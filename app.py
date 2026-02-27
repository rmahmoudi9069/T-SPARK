# ===================== app.py =====================
from flask import Flask, render_template, request, jsonify, send_file, url_for, send_from_directory, redirect, session
import os
import geopandas as gpd
import io, pandas as pd
import folium
from folium.plugins import Fullscreen
import numpy as np
import branca.colormap as cm
from functools import partial
import fiona
from shapely.geometry import LineString
import datetime
from shapely.ops import unary_union
from shapely.geometry import Point, Polygon, MultiPolygon
from folium.plugins import Fullscreen, MarkerCluster
import pulp
from pulp import LpProblem, LpMinimize, LpMaximize, LpVariable, lpSum, PULP_CBC_CMD, LpStatus
import networkx as nx
import osmnx as ox
import partridge as ptg
from peartree import load_feed_as_graph
import osmnx._osm_xml as _osm_xml
from lxml import etree
import logging
from pyrosm import OSM
import r5py 
import zipfile
import uuid
import shutil
import tempfile
import glob
from fiona import Env
import time
import gc
import stat



app = Flask(__name__)

app.secret_key = "some_random_string"  # needed if you prefer session
user_paths = {
    "eff_file": None,
    "mpi_file": None,
    "eff_change_file": None,
    "tech_change_file": None
}

# --------- Style Functions for Map Rendering (Demographic) ----------
def style_func(feature, selected_criterion, colormap):
    try:
        val = float(feature["properties"].get(selected_criterion, None))
    except (TypeError, ValueError):
        val = None
    if val is None or val == 0 or np.isnan(val):
        return {"fillColor": "transparent", "color": "black", "weight": 1, "fillOpacity": 0.3}
    else:
        return {"fillColor": colormap(val), "color": "black", "weight": 1, "fillOpacity": 0.7}

def style_func_boundary(feature):
    return {"fillColor": "transparent", "color": "black", "weight": 1, "fillOpacity": 0.3}

def highlight_func(feature):
    return {"weight": 2, "color": "blue"}

# --------- Demographic Map Creation Function ----------
def create_demographic_map(merged, selected_criterion, selected_zones_tuple, base_map="OpenStreetMap"):
    if selected_criterion.strip() != "" and selected_criterion.lower() != "0-taz only":
        merged.loc[~merged["gta06"].isin(selected_zones_tuple), selected_criterion] = np.nan

    center_lat = merged.geometry.centroid.y.mean()
    center_lon = merged.geometry.centroid.x.mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles=base_map)
    Fullscreen(position='topright').add_to(m)

    if selected_criterion.strip() == "" or selected_criterion.lower() == "0-taz only":
        folium.GeoJson(
            data=merged.to_json(),
            style_function=style_func_boundary,
            highlight_function=highlight_func,
            tooltip=folium.GeoJsonTooltip(
                fields=["gta06"],
                aliases=["TAZ ID"],
                localize=True
            )
        ).add_to(m)
    else:
        data_col = merged[selected_criterion]
        valid_data = data_col[(data_col.notna()) & (data_col != 0)]
        if len(valid_data) > 0:
            vmin, vmax = valid_data.min(), valid_data.max()
            colormap = cm.LinearColormap(
                colors=["red", "orange", "yellow", "green", "darkgreen"],
                vmin=vmin, vmax=vmax
            )
            colormap.caption = selected_criterion
            style_function_partial = partial(style_func, selected_criterion=selected_criterion, colormap=colormap)
            folium.GeoJson(
                data=merged.to_json(),
                style_function=style_function_partial,
                highlight_function=highlight_func,
                tooltip=folium.GeoJsonTooltip(
                    fields=["gta06", selected_criterion],
                    aliases=["TAZ ID", selected_criterion],
                    localize=True
                )
            ).add_to(m)
            colormap.add_to(m)
            custom_css = """
            <style>
            .branca .colormap {
                position: absolute !important;
                top: 10px !important;
                left: 50% !important;
                transform: translateX(-50%) !important;
                width: 300px !important;
                z-index: 9999 !important;
            }
            </style>
            """
            m.get_root().header.add_child(folium.Element(custom_css))
        else:
            folium.GeoJson(
                data=merged.to_json(),
                style_function=style_func_boundary,
                highlight_function=highlight_func,
                tooltip=folium.GeoJsonTooltip(
                    fields=["gta06", selected_criterion],
                    aliases=["TAZ ID", selected_criterion],
                    localize=True
                )
            ).add_to(m)
    return m

# -------------------------------
# New Helper Functions for Adjacent Network Analysis
# -------------------------------
def load_gtfs_data(gtfs_folder):
    routes_path = os.path.join(gtfs_folder, "routes.txt")
    trips_path = os.path.join(gtfs_folder, "trips.txt")
    shapes_path = os.path.join(gtfs_folder, "shapes.txt")
    if not (os.path.exists(routes_path) and os.path.exists(trips_path) and os.path.exists(shapes_path)):
        raise Exception("One or more GTFS files (routes.txt, trips.txt, shapes.txt) not found in the specified folder.")
    routes = pd.read_csv(routes_path)
    trips = pd.read_csv(trips_path)
    shapes = pd.read_csv(shapes_path)
    return routes, trips, shapes

# -------------------------------
# New Helper Function for Performance GTFS Data Loading
# -------------------------------
def load_gtfs_data_performance(gtfs_folder):
    """
    Load GTFS files specifically for the performance evaluation tool.
    Ensures that the 'route_id' is read as a string.
    """
    routes_path = os.path.join(gtfs_folder, "routes.txt")
    trips_path = os.path.join(gtfs_folder, "trips.txt")
    shapes_path = os.path.join(gtfs_folder, "shapes.txt")
    if not (os.path.exists(routes_path) and os.path.exists(trips_path) and os.path.exists(shapes_path)):
        raise Exception("One or more required GTFS files (routes.txt, trips.txt, shapes.txt) not found in the specified folder.")
    routes = pd.read_csv(routes_path, dtype={'route_id': str})
    trips = pd.read_csv(trips_path, dtype={'trip_id': str, 'service_id': str, 'route_id': str})
    shapes = pd.read_csv(shapes_path)
    return routes, trips, shapes


def process_gtfs_data(routes, trips, shapes):
    trips_routes = pd.merge(trips, routes, on='route_id', how='left')
    shapes_trips = pd.merge(shapes, trips_routes, on='shape_id', how='left')
    shapes_trips = shapes_trips.dropna(subset=['shape_pt_lat', 'shape_pt_lon'])
    gdf_routes = shapes_trips.sort_values('shape_pt_sequence').groupby(
        ['shape_id', 'route_id', 'route_short_name', 'route_long_name', 'route_color']
    ).apply(lambda x: LineString(zip(x['shape_pt_lon'], x['shape_pt_lat']))).reset_index(name='geometry')
    gdf_routes = gpd.GeoDataFrame(gdf_routes, geometry='geometry')
    gdf_routes['geometry'] = gdf_routes['geometry'].simplify(tolerance=0.001, preserve_topology=True)
    gdf_routes.set_crs(epsg=4326, inplace=True)
    gdf_routes['display_name'] = gdf_routes.apply(
        lambda row: f"Route {row['route_id']} - {row['route_short_name']} : {row['route_long_name']}", axis=1
    )
    return gdf_routes

def add_transit_layer(m, transit_gdf, selected_routes, route_colors):
    # Assumes selected_routes and route_colors are lists of equal length.
    for rid, col in zip(selected_routes, route_colors):
        filtered = transit_gdf[transit_gdf['route_id'].astype(str) == rid]
        for _, row in filtered.iterrows():
            folium.GeoJson(
                row['geometry'].__geo_interface__,
                style_function=lambda feature, c=col: {"color": c, "weight": 4, "opacity": 0.8},
                tooltip=row['display_name']
            ).add_to(m)
    return m

def create_adjacent_map(zones, transit_gdf, under_eval_ids, under_eval_colors, adjacent_ids, adjacent_colors, base_map="OpenStreetMap"):
    m = folium.Map(location=[43.6532, -79.3832], zoom_start=10, tiles=base_map)
    Fullscreen(position='topright').add_to(m)
    if zones is not None:
        folium.GeoJson(
            data=zones.to_json(),
            style_function=lambda feature: {"fillColor": "transparent", "color": "black", "weight": 1, "fillOpacity": 0.3},
            tooltip=folium.GeoJsonTooltip(fields=["gta06"], aliases=["Zone ID"], localize=True)
        ).add_to(m)
    if under_eval_ids:
        m = add_transit_layer(m, transit_gdf, under_eval_ids, under_eval_colors)
    if adjacent_ids:
        m = add_transit_layer(m, transit_gdf, adjacent_ids, adjacent_colors)
    return m

# -------------------------------
# New Endpoint for GTFS Data Reading
# -------------------------------
@app.route("/read_gtfs", methods=["POST"])
def read_gtfs():
    gtfs_path = request.form.get("gtfs_path")
    if not gtfs_path:
        return jsonify({"error": "No GTFS path provided"}), 400
    try:
        routes, trips, shapes = load_gtfs_data(gtfs_path)
        transit_gdf = process_gtfs_data(routes, trips, shapes)
        route_options = transit_gdf[['route_id', 'display_name']].drop_duplicates().to_dict(orient='records')
        return jsonify(route_options)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------
# New Endpoint for GTFS Data Reading for Coverage Analysis
# -------------------------------
@app.route("/read_gtfs_coverage", methods=["POST"])
def read_gtfs_coverage():
    gtfs_path = request.form.get("gtfs_path")
    if not gtfs_path:
        return jsonify({"error": "GTFS data path not provided."}), 400
    try:
        # Load only the files needed for coverage analysis
        stops = pd.read_csv(os.path.join(gtfs_path, "stops.txt"), dtype={'stop_id': str})
        routes = pd.read_csv(os.path.join(gtfs_path, "routes.txt"), dtype={'route_id': str})
        calendar_dates = pd.read_csv(os.path.join(gtfs_path, "calendar_dates.txt"), dtype={'date': str})
        
        # Determine available date range from calendar_dates.txt
        start_date = calendar_dates['date'].min()
        end_date = calendar_dates['date'].max()
        
        # Prepare a list of transit routes (combine short and long names if available)
        if 'route_short_name' in routes.columns and 'route_long_name' in routes.columns:
            routes["display_name"] = routes["route_short_name"].astype(str) + " - " + routes["route_long_name"].astype(str)
        else:
            routes["display_name"] = routes["route_id"]
        route_options = routes[['route_id', 'display_name']].drop_duplicates().to_dict(orient='records')
        
        return jsonify({
            "start_date": start_date,
            "end_date": end_date,
            "routes": route_options
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500





# --------- Main Routes ----------
@app.route("/")
def home():
    return render_template("index.html")


# --------- demographic ----------
@app.route("/demographic", methods=["GET", "POST"])
def demographic():
    if request.method == "POST":
        shapefile_path = request.form.get("shapefile")
        excel_path = request.form.get("excel")
        if not shapefile_path:
            return "Error: Please enter a file path for the shapefile (.shp)."

        unit_type = request.form.get("unit_type")
        selected_criterion = request.form.get("criterion") or ""
        base_map = request.form.get("base_map")
        min_unit = request.form.get("min_unit") or ""
        max_unit = request.form.get("max_unit") or ""

        try:
            with fiona.Env(SHAPE_RESTORE_SHX="YES"):
                gdf = gpd.read_file(shapefile_path, driver="ESRI Shapefile")
        except Exception as e:
            return f"Error reading shapefile: {e}"
        if "gta06" not in gdf.columns:
            return "Error: Shapefile must contain a 'gta06' column."
        gdf["gta06"] = gdf["gta06"].astype(int)
        gdf = gdf.to_crs(epsg=4326)
        if min_unit.strip() != "" and max_unit.strip() != "":
            try:
                min_val = int(min_unit)
                max_val = int(max_unit)
                gdf = gdf[(gdf["gta06"] >= min_val) & (gdf["gta06"] <= max_val)].copy()
            except ValueError:
                pass
        if excel_path.strip() != "":
            try:
                df = pd.read_excel(excel_path)
            except Exception as e:
                return f"Error reading Excel file: {e}"
            merged = gdf.merge(df, on="gta06", how="left")
        else:
            merged = gdf.copy()
        selected_zones_tuple = tuple(merged["gta06"].unique())
        m = create_demographic_map(merged, selected_criterion, selected_zones_tuple, base_map=base_map)
        map_html = m._repr_html_()
        return render_template("demographic.html", map_html=map_html, container_class="",
                               prev_unit_type=unit_type,
                               prev_min_unit=min_unit,
                               prev_max_unit=max_unit,
                               prev_criterion=selected_criterion,
                               prev_shapefile=shapefile_path,
                               prev_excel=excel_path,
                               prev_basemap=base_map)
    else:
        default_map = folium.Map(location=[43.6532, -79.3832], zoom_start=10, tiles="OpenStreetMap")
        Fullscreen(position='topright').add_to(default_map)
        map_html = default_map._repr_html_()
        return render_template("demographic.html", map_html=map_html, container_class="",
                               prev_unit_type="TAZ",
                               prev_min_unit="",
                               prev_max_unit="",
                               prev_criterion="",
                               prev_shapefile="",
                               prev_excel="",
                               prev_basemap="OpenStreetMap")

@app.route("/adjacent", methods=["GET", "POST"])
def adjacent():
    if request.method == "POST":
        include_zones = request.form.get("include_zones")
        zones = None
        if include_zones == "yes":
            zone_shapefile = request.form.get("zone_shapefile")
            try:
                zones = gpd.read_file(zone_shapefile).to_crs(epsg=4326)
                zones["gta06"] = zones["gta06"].astype(int)
                # **Added filtering for zone IDs:**
                zone_min_unit = request.form.get("zone_min_unit") or ""
                zone_max_unit = request.form.get("zone_max_unit") or ""
                if zone_min_unit.strip() != "" and zone_max_unit.strip() != "":
                    try:
                        min_val = int(zone_min_unit)
                        max_val = int(zone_max_unit)
                        zones = zones[(zones["gta06"] >= min_val) & (zones["gta06"] <= max_val)].copy()
                    except ValueError:
                        pass
            except Exception as e:
                return f"Error reading zone shapefile: {e}"

        gtfs_path = request.form.get("gtfs_path")
        try:
            routes, trips, shapes = load_gtfs_data(gtfs_path)
            transit_gdf = process_gtfs_data(routes, trips, shapes)
        except Exception as e:
            return f"Error processing GTFS data: {e}"

        under_eval_ids = request.form.getlist("under_eval_routes")
        adjacent_ids = request.form.getlist("adjacent_routes")

        # Retrieve colors correctly as single values (radio buttons)
        under_eval_color = request.form.get("under_eval_color", "Blue")  # default to Blue
        adjacent_color = request.form.get("adjacent_color", "Red")       # default to Red

        # Construct color lists (same color for all selected routes)
        under_eval_colors = [under_eval_color] * len(under_eval_ids)
        adjacent_colors = [adjacent_color] * len(adjacent_ids)

        base_map = request.form.get("base_map")

        m = create_adjacent_map(zones, transit_gdf, under_eval_ids, under_eval_colors, adjacent_ids, adjacent_colors, base_map=base_map)
        map_html = m._repr_html_()
        return render_template("adjacent.html", map_html=map_html,
                               prev_include_zones=include_zones,
                               prev_gtfs_path=gtfs_path,
                               prev_under_eval_routes=under_eval_ids,
                               prev_under_eval_colors=under_eval_colors,
                               prev_adjacent_routes=adjacent_ids,
                               prev_adjacent_colors=adjacent_colors,
                               prev_basemap=base_map)
    else:
        default_map = folium.Map(location=[43.6532, -79.3832], zoom_start=10, tiles="OpenStreetMap")
        Fullscreen(position='topright').add_to(default_map)
        map_html = default_map._repr_html_()
        return render_template("adjacent.html", map_html=map_html,
                               prev_include_zones="no",
                               prev_gtfs_path="",
                               prev_under_eval_routes=[],
                               prev_under_eval_colors=[],
                               prev_adjacent_routes=[],
                               prev_adjacent_colors=[],
                               prev_basemap="OpenStreetMap")


# -------------------------------
# Coverage Analysis Route
# -------------------------------
@app.route("/coverage", methods=["GET", "POST"])
def coverage():
    map_html = ""
    dates_range = ""
    message = ""

    if request.method == "POST":
        gtfs_path = request.form.get("gtfs_path")
        boundary_path = request.form.get("boundary_shapefile")
        buffer_distance = float(request.form.get("buffer_distance", 400))
        selected_date_str = request.form.get("selected_date")
        analysis_type = request.form.get("analysis_type")
        selected_routes = request.form.getlist("selected_routes")
        operation_period = request.form.get("operation_period", "Weekend")
        base_map = request.form.get("base_map", "OpenStreetMap")

        try:
            boundary_gdf = gpd.read_file(boundary_path).to_crs(epsg=4326)

            # Load GTFS files required for coverage analysis
            stops_df = pd.read_csv(os.path.join(gtfs_path, "stops.txt"), dtype={'stop_id': str})
            stop_times_df = pd.read_csv(os.path.join(gtfs_path, "stop_times.txt"), dtype={'stop_id': str, 'trip_id': str})
            trips_df = pd.read_csv(os.path.join(gtfs_path, "trips.txt"), dtype={'trip_id': str, 'route_id': str, 'service_id': str})
            calendar_dates_df = pd.read_csv(os.path.join(gtfs_path, "calendar_dates.txt"), dtype={'date': str, 'service_id': str})

            dates_range = f"{calendar_dates_df['date'].min()} to {calendar_dates_df['date'].max()}"

            # Parse the selected date (assumed format is YYYY-MM-DD from the date input)
            selected_date = datetime.datetime.strptime(selected_date_str, "%Y-%m-%d")
            
            # IMPORTANT: Format the date as YYYYMMDD to match GTFS file format
            formatted_date = selected_date.strftime('%Y%m%d')

            # Define operation periods
            periods = {
                "Weekend": (datetime.time(0, 0), datetime.time(23, 59)),
                "Weekday AM Peak": (datetime.time(6, 0), datetime.time(10, 0)),
                "Weekday Mid-day": (datetime.time(10, 0), datetime.time(15, 0)),
                "Weekday PM Peak": (datetime.time(15, 0), datetime.time(19, 0)),
                "Weekday After PM Peak": (datetime.time(19, 0), datetime.time(23, 59)),
                "Weekday Overnight": (datetime.time(0, 0), datetime.time(6, 0))
            }

            # Get active services using the correctly formatted date
            active_services = calendar_dates_df[
                calendar_dates_df['date'] == formatted_date
            ]['service_id'].unique()

            # Get active trips DataFrame then extract trip IDs
            active_trips_df = trips_df[trips_df['service_id'].isin(active_services)]
            active_trip_ids = active_trips_df['trip_id'].unique()

            # For catchment area analysis, if selected routes are provided, filter trips accordingly
            if analysis_type == "catchment_area" and selected_routes:
                route_trips_df = trips_df[trips_df['route_id'].isin(selected_routes)]
                active_trip_ids = route_trips_df['trip_id'].unique()

            # Get active stop IDs from stop_times based on active trip IDs
            active_stop_ids = stop_times_df[
                stop_times_df['trip_id'].isin(active_trip_ids)
            ]['stop_id'].unique()

            active_stops = stops_df[stops_df['stop_id'].isin(active_stop_ids)]
            if active_stops.empty:
                message = "No active stops found for the selected date/analysis."
            else:
                active_stops_gdf = gpd.GeoDataFrame(
                    active_stops,
                    geometry=gpd.points_from_xy(active_stops.stop_lon, active_stops.stop_lat),
                    crs="EPSG:4326"
                )

                # Compute the coverage area (buffer around active stops)
                coverage_geom = active_stops_gdf.to_crs(epsg=32617).buffer(buffer_distance).to_crs(epsg=4326).unary_union
                coverage_gdf = gpd.GeoDataFrame(geometry=[coverage_geom], crs="EPSG:4326")

                m = folium.Map(location=[boundary_gdf.geometry.centroid.y.mean(), boundary_gdf.geometry.centroid.x.mean()],
                               zoom_start=11, tiles=base_map)
                Fullscreen(position='topright').add_to(m)

                # Always add the city boundary
                folium.GeoJson(boundary_gdf.to_json(), style_function=lambda x: {'color': 'black', 'fillOpacity': 0}).add_to(m)

                if analysis_type == "transit_deserts":
                    # Calculate transit deserts: city boundary minus coverage area
                    desert_geom = boundary_gdf.geometry.unary_union.difference(coverage_geom)
                    desert_gdf = gpd.GeoDataFrame(geometry=[desert_geom], crs="EPSG:4326")
                    folium.GeoJson(desert_gdf.to_json(), style_function=lambda x: {
                        'color': 'red', 'fillColor': 'red', 'fillOpacity': 0.5}).add_to(m)
                else:
                    folium.GeoJson(coverage_gdf.to_json(), style_function=lambda x: {
                        'color': 'blue', 'fillColor': 'blue', 'fillOpacity': 0.5}).add_to(m)

                map_html = m._repr_html_()
                message = "Map generated successfully."

        except Exception as e:
            message = f"Error: {str(e)}"

    else:
        # Default map (Toronto centered)
        m = folium.Map(location=[43.6532, -79.3832], zoom_start=11, tiles="OpenStreetMap")
        Fullscreen(position='topright').add_to(m)
        map_html = m._repr_html_()

    return render_template("coverage.html", map_html=map_html, dates_range=dates_range, message=message)





# -------------------------------
# New Endpoint for Reading Route-Level Efficiency Scores
# -------------------------------
@app.route("/read_efficiency", methods=["POST"])
def read_efficiency():
    # This endpoint is specific for route-level analysis.
    eff_path = request.form.get("eff_path")
    if not eff_path:
        return jsonify({"error": "Efficiency file path not provided."}), 400
    try:
        # Read the Excel file; we expect the first column to be route_id
        eff_df = pd.read_excel(eff_path)
        if 'route_id' not in eff_df.columns:
            return jsonify({"error": "The efficiency file must contain a 'route_id' column."}), 400
        # All other columns are assumed to be evaluation period headers
        evaluation_periods = [col for col in eff_df.columns if col != 'route_id']
        return jsonify({"evaluation_periods": evaluation_periods})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/read_efficiency_segment", methods=["POST"])
def read_efficiency_segment():
    seg_path = request.form.get("eff_seg_path")
    if not seg_path:
        return jsonify({"error": "Segment efficiency file path not provided."}), 400
    try:
        # Specify the engine explicitly
        eff_df = pd.read_excel(seg_path, engine="openpyxl")
        eff_df.columns = eff_df.columns.str.strip()

        # Must have 'route_id' and 'geometry' columns
        if 'route_id' not in eff_df.columns or 'geometry' not in eff_df.columns:
            return jsonify({"error": "Segment file must contain 'route_id' and 'geometry' columns."}), 400

        # Exclude route_id and geometry if present
        skip_cols = ['route_id', 'geometry']
        evaluation_periods = [c for c in eff_df.columns if c not in skip_cols]

        return jsonify({"evaluation_periods": evaluation_periods})
    except Exception as e:
        return jsonify({"error": str(e)}), 500





# -------------------------------
# New Endpoint for Reading GTFS Data for Performance Evaluation Tool
# -------------------------------
@app.route("/read_gtfs_performance", methods=["POST"])
def read_gtfs_performance():
    # This endpoint reads GTFS files specifically for the performance evaluation tool.
    gtfs_path = request.form.get("gtfs_path")
    if not gtfs_path:
        return jsonify({"error": "GTFS data path not provided."}), 400
    try:
        # Define paths to GTFS files
        routes_path = os.path.join(gtfs_path, "routes.txt")
        trips_path = os.path.join(gtfs_path, "trips.txt")
        shapes_path = os.path.join(gtfs_path, "shapes.txt")
        if not (os.path.exists(routes_path) and os.path.exists(trips_path) and os.path.exists(shapes_path)):
            return jsonify({"error": "One or more required GTFS files (routes.txt, trips.txt, shapes.txt) not found."}), 400
        # Load the necessary files with proper data types
        routes = pd.read_csv(routes_path, dtype={'route_id': str})
        trips = pd.read_csv(trips_path, dtype={'trip_id': str, 'service_id': str, 'route_id': str})
        shapes = pd.read_csv(shapes_path)
        # Use the existing processing function to get route geometries
        transit_gdf = process_gtfs_data(routes, trips, shapes)
        # Ensure route_id is treated as string (for merging consistency)
        transit_gdf['route_id'] = transit_gdf['route_id'].astype(str)
        # Prepare a list of routes (with display names) for the performance tool
        route_options = transit_gdf[['route_id', 'display_name']].drop_duplicates().to_dict(orient='records')
        return jsonify({"routes": route_options})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



# -------------------------------
# New Endpoint for Performance Evaluation Tool
# -------------------------------

@app.route("/performance", methods=["GET", "POST"])
def performance():
    map_html = ""
    message = ""
    if request.method == "POST":
        # 1) Read user input from the form
        analysis_level = request.form.get("analysis_level")  # "route_level" or "segment_level"
        include_boundary = request.form.get("include_boundary")  # "yes" or "no"
        boundary_path = request.form.get("boundary_shapefile")  # City boundary shapefile
        base_map = request.form.get("base_map", "OpenStreetMap")

        # Efficiency and other files
        eff_file = request.form.get("eff_file")         # Route-level efficiency
        eff_file_seg = request.form.get("eff_file_seg") # Segment-level efficiency
        mpi_file = request.form.get("mpi_file")
        eff_change_file = request.form.get("eff_change_file")
        tech_change_file = request.form.get("tech_change_file")

        # Save them globally so the new endpoints can read them:
        global user_paths
        user_paths["eff_file"] = eff_file
        user_paths["mpi_file"] = mpi_file
        user_paths["eff_change_file"] = eff_change_file
        user_paths["tech_change_file"] = tech_change_file        

        gtfs_path = request.form.get("gtfs_path_route")  # GTFS path for route-level analysis
        eval_period = request.form.get("evaluation_period")

        try:
            # 2) If user chose "yes" for city boundary, read it; else skip
            boundary_gdf = None
            if include_boundary == "yes" and boundary_path:
                boundary_gdf = gpd.read_file(boundary_path).to_crs(epsg=4326)

            # 3) Distinguish analysis level
            if analysis_level == "route_level":
                # Read the route-level efficiency file
                eff_df = pd.read_excel(eff_file)
                eff_df['route_id'] = eff_df['route_id'].astype(str)

                if eval_period not in eff_df.columns:
                    message = f"Evaluation period '{eval_period}' not found in the efficiency file."
                    return render_template("performance.html", map_html="", message=message)

                # Read GTFS data for route geometries
                routes, trips, shapes = load_gtfs_data_performance(gtfs_path)
                transit_gdf = process_gtfs_data(routes, trips, shapes)
                transit_gdf['route_id'] = transit_gdf['route_id'].astype(str)

                # Merge with efficiency data
                merged_df = pd.merge(transit_gdf, eff_df, on="route_id", how="left")

                # Read user-selected routes
                selected_routes = request.form.getlist("selected_routes")
                if selected_routes and "all" not in selected_routes:
                    merged_df = merged_df[merged_df["route_id"].isin(selected_routes)]

                # Keep only rows with non-null efficiency
                merged_df = merged_df[merged_df[eval_period].notnull()]
                if merged_df.empty:
                    message = "No routes with efficiency scores found for the selected evaluation period."
                    return render_template("performance.html", map_html="", message=message)

                # -- B: Replace the old 2-color scheme with your multi-stop color scale --
                from branca.colormap import LinearColormap
                efficiency_colormap = LinearColormap(
                    colors=["black", "#8B0000", "#FF0000", "orange", "yellow",
                            "#90EE90", "#00FF00", "#008000", "#006400"],
                    index=[0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                    vmin=0, vmax=1
                )
                efficiency_colormap.caption = f"Efficiency ({eval_period})"

                # Assign color based on efficiency score
                merged_df["color"] = merged_df[eval_period].apply(lambda val: efficiency_colormap(val))

                # Build map
                if boundary_gdf is not None:
                    center_lat = boundary_gdf.geometry.centroid.y.mean()
                    center_lon = boundary_gdf.geometry.centroid.x.mean()
                else:
                    center_lat, center_lon = 43.6532, -79.3832  # Toronto fallback

                m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles=base_map)
                Fullscreen(position='topright').add_to(m)

                # If boundary_gdf was read, add it
                if boundary_gdf is not None:
                    folium.GeoJson(
                        boundary_gdf.to_json(),
                        style_function=lambda x: {'color': 'black', 'fillOpacity': 0}
                    ).add_to(m)

                # Plot each route
                for idx, row in merged_df.iterrows():
                    if row['geometry'] is not None:
                        coords = [(pt[1], pt[0]) for pt in row['geometry'].coords]
                        popup_text = f"Route {row['route_id']}<br>{eval_period}: {row[eval_period]:.3f}"
                        folium.PolyLine(
                            locations=coords,
                            color=row['color'],
                            weight=5,
                            opacity=0.8,
                            popup=popup_text
                        ).add_to(m)

                efficiency_colormap.add_to(m)
                map_html = m._repr_html_()
                message = "Map generated successfully for Route Level analysis."

            elif analysis_level == "segment_level":
                # Segment-level
                eff_df = pd.read_excel(eff_file_seg, engine="openpyxl")
                if 'geometry' not in eff_df.columns:
                    message = "Efficiency file does not contain a 'geometry' column."
                    return render_template("performance.html", map_html="", message=message)
                # Use shapely.wkt.loads to convert valid WKT strings to geometry objects
                from shapely import wkt
                def safe_wkt_load(x):
                    # Only try loading if x is a non-empty string
                    s = str(x).strip() if pd.notnull(x) else ""
                    if s == "":
                        return None
                    try:
                        return wkt.loads(s)
                    except Exception as e:
                        # Optionally log e; returning None skips problematic rows
                        return None

                eff_df["geometry"] = eff_df["geometry"].apply(safe_wkt_load)

                segment_gdf = gpd.GeoDataFrame(eff_df, geometry="geometry", crs="EPSG:4326")

                if eval_period not in segment_gdf.columns:
                    message = f"Evaluation period '{eval_period}' not found in the efficiency file."
                    return render_template("performance.html", map_html="", message=message)

                segment_gdf = segment_gdf[segment_gdf[eval_period].notnull()]
                if segment_gdf.empty:
                    message = "No segments with efficiency scores found for the selected evaluation period."
                    return render_template("performance.html", map_html="", message=message)


                # NEW: Convert the evaluation period column to numeric so comparisons work correctly.
                segment_gdf[eval_period] = pd.to_numeric(segment_gdf[eval_period], errors='coerce')

                # Compute vmin and vmax from your data
                vmin = segment_gdf[eval_period].min()
                vmax = segment_gdf[eval_period].max()


                # Ensure vmin is less than vmax: if not, swap them.
                if vmin > vmax:
                    vmin, vmax = vmax, vmin


                # If all values are identical or reversed, vmin >= vmax => expand slightly
                if vmin >= vmax:
                    # Expand the range a bit so we don't get thresholds that are all the same
                    vmin -= 0.0001
                    vmax += 0.0001


                # Scale your thresholds relative to vmin and vmax.
                thresholds = [vmin + (vmax - vmin) * t for t in [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]]


                from branca.colormap import LinearColormap
                efficiency_colormap = LinearColormap(
                    colors=["black", "#8B0000", "#FF0000", "orange", "yellow",
                            "#90EE90", "#00FF00", "#008000", "#006400"],
                    index=thresholds,
                    vmin=vmin, vmax=vmax
                )
                efficiency_colormap.caption = f"Efficiency ({eval_period})"

                segment_gdf["color"] = segment_gdf[eval_period].apply(lambda val: efficiency_colormap(val))

                if boundary_gdf is not None:
                    center_lat = boundary_gdf.geometry.centroid.y.mean()
                    center_lon = boundary_gdf.geometry.centroid.x.mean()
                else:
                    center_lat, center_lon = 43.6532, -79.3832

                m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles=base_map)
                Fullscreen(position='topright').add_to(m)

                if boundary_gdf is not None:
                    folium.GeoJson(
                        boundary_gdf.to_json(),
                        style_function=lambda x: {'color': 'black', 'fillOpacity': 0}
                    ).add_to(m)

                for idx, row in segment_gdf.iterrows():
                    if row['geometry'] is not None:
                        coords = [(pt[1], pt[0]) for pt in row['geometry'].coords]
                        segment_id = row.get("route_id", f"Segment {idx}")
                        popup_text = f"Segment: {segment_id}<br>{eval_period}: {row[eval_period]:.3f}"
                        folium.PolyLine(
                            locations=coords,
                            color=row['color'],
                            weight=4,
                            opacity=0.8,
                            popup=popup_text
                        ).add_to(m)

                efficiency_colormap.add_to(m)
                map_html = m._repr_html_()
                message = "Map generated successfully for Segment Level analysis."

            else:
                message = "Invalid analysis level selected."

        except Exception as e:
            message = f"Error: {str(e)}"

    else:
        # GET request: show a blank default map
        m = folium.Map(location=[43.6532, -79.3832], zoom_start=11, tiles="OpenStreetMap")
        Fullscreen(position='topright').add_to(m)
        map_html = m._repr_html_()

    return render_template("performance.html", map_html=map_html, message=message)


###############################################################################
#  New Endpoints for Additional Buttons
###############################################################################

@app.route("/get_performance_routes", methods=["GET"])
def get_performance_routes():
    """
    Returns a list of (route_id, display_name) in JSON format 
    so the front-end can build checkboxes.
    Similar to read_gtfs_performance, but simpler.
    """
    try:
        # Suppose we read the same GTFS data used in route-level analysis:
        gtfs_path = r"C:\path\to\some\default\gtfs_folder"  # or fetch from config
        routes_df = pd.read_csv(os.path.join(gtfs_path, "routes.txt"), dtype={'route_id': str})
        # We build display_name from route_short_name + route_long_name if possible
        if "route_short_name" in routes_df.columns and "route_long_name" in routes_df.columns:
            routes_df["display_name"] = routes_df["route_short_name"].astype(str) + " - " + routes_df["route_long_name"].astype(str)
        else:
            routes_df["display_name"] = routes_df["route_id"]

        # Convert to list of dict
        route_options = routes_df[["route_id","display_name"]].drop_duplicates().to_dict(orient='records')
        return jsonify({"routes": route_options})
    except Exception as e:
        return jsonify({"error": str(e)}), 500








@app.route("/list_trend_routes_json", methods=["POST"])
def list_trend_routes_json():
    """
    Reads the posted JSON: {"excelPath": "..."}.
    Opens that file, strips column headers, extracts route IDs, returns them in JSON.
    """
    data = request.get_json()
    eff_path = data.get("excelPath", "").strip()
    if not eff_path or not os.path.exists(os.path.normpath(eff_path)):
        return jsonify({"routes": []})
    try:
        eff_path = os.path.normpath(eff_path)
        df = pd.read_excel(eff_path)
        df.columns = df.columns.str.strip()  # remove extra spaces from headers
        if "route_id" not in df.columns:
            return jsonify({"routes": []})
        df["route_id"] = df["route_id"].astype(str)
        df["display_name"] = "Route " + df["route_id"]
        route_list = df[["route_id", "display_name"]].drop_duplicates().to_dict(orient="records")
        return jsonify({"routes": route_list})
    except Exception as e:
        return jsonify({"routes": [], "error": str(e)})


@app.route("/list_mpi_routes_json", methods=["POST"])
def list_mpi_routes_json():
    data = request.get_json()
    mpi_path = data.get("excelPath", "").strip()
    if not mpi_path or not os.path.exists(os.path.normpath(mpi_path)):
        return jsonify({"routes": []})
    try:
        mpi_path = os.path.normpath(mpi_path)
        df = pd.read_excel(mpi_path)
        df.columns = df.columns.str.strip()
        if "route_id" not in df.columns:
            return jsonify({"routes": []})
        df["route_id"] = df["route_id"].astype(str)
        df["display_name"] = "Route " + df["route_id"]
        route_list = df[["route_id", "display_name"]].drop_duplicates().to_dict(orient="records")
        return jsonify({"routes": route_list})
    except Exception as e:
        return jsonify({"routes": [], "error": str(e)})


@app.route("/list_eff_change_routes_json", methods=["POST"])
def list_eff_change_routes_json():
    data = request.get_json()
    excel_path = data.get("excelPath", "").strip()
    if not excel_path or not os.path.exists(os.path.normpath(excel_path)):
        return jsonify({"routes": []})
    try:
        excel_path = os.path.normpath(excel_path)
        df = pd.read_excel(excel_path)
        df.columns = df.columns.str.strip()
        if "route_id" not in df.columns:
            return jsonify({"routes": []})
        df["route_id"] = df["route_id"].astype(str)
        df["display_name"] = "Route " + df["route_id"]
        route_list = df[["route_id", "display_name"]].drop_duplicates().to_dict(orient="records")
        return jsonify({"routes": route_list})
    except Exception as e:
        return jsonify({"routes": [], "error": str(e)})


@app.route("/list_tech_change_routes_json", methods=["POST"])
def list_tech_change_routes_json():
    data = request.get_json()
    excel_path = data.get("excelPath", "").strip()
    if not excel_path or not os.path.exists(os.path.normpath(excel_path)):
        return jsonify({"routes": []})
    try:
        excel_path = os.path.normpath(excel_path)
        df = pd.read_excel(excel_path)
        df.columns = df.columns.str.strip()
        if "route_id" not in df.columns:
            return jsonify({"routes": []})
        df["route_id"] = df["route_id"].astype(str)
        df["display_name"] = "Route " + df["route_id"]
        route_list = df[["route_id", "display_name"]].drop_duplicates().to_dict(orient="records")
        return jsonify({"routes": route_list})
    except Exception as e:
        return jsonify({"routes": [], "error": str(e)})






@app.route("/analyze_trend", methods=["POST"])
def analyze_trend():
    """
    Reads "excelPath" and selected "routes" from JSON.
    Returns { x: [...], series: { route_id: [scores, ...] } } for a Plotly line chart.
    """
    data = request.get_json()
    excel_path = data.get("excelPath", "").strip()
    selected_routes = data.get("routes", [])
    if not excel_path or not os.path.exists(os.path.normpath(excel_path)):
        return jsonify({"error": "Trend file not found"}), 400
    try:
        excel_path = os.path.normpath(excel_path)
        df_raw = pd.read_excel(excel_path)
        df_raw.columns = df_raw.columns.str.strip()
        if "route_id" not in df_raw.columns:
            return jsonify({"error": "No 'route_id' column found"}), 400
        if "Average efficiency" in df_raw.columns:
            df_raw.drop(columns=["Average efficiency"], inplace=True)
        melt = df_raw.melt(
            id_vars=["route_id"],
            var_name="PeriodCol",
            value_name="EffScore"
        )
        melt["route_id"] = melt["route_id"].astype(str)
        if selected_routes:
            melt = melt[melt["route_id"].isin(selected_routes)]
        periods = [c for c in df_raw.columns if c != "route_id"]
        route_series = {}
        for r in melt["route_id"].unique():
            sub = melt[melt["route_id"] == r]
            pmap = dict(zip(sub["PeriodCol"], sub["EffScore"]))
            route_series[r] = [pmap.get(p, None) for p in periods]
        return jsonify({"x": list(periods), "series": route_series})
    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.route("/visualize_mpi", methods=["POST"])
def visualize_mpi():
    data = request.get_json()
    excel_path = data.get("excelPath", "").strip()
    selected_routes = data.get("routes", [])
    if not excel_path or not os.path.exists(os.path.normpath(excel_path)):
        return jsonify({"error": "MPI file not found"}), 400
    try:
        excel_path = os.path.normpath(excel_path)
        df = pd.read_excel(excel_path)
        df.columns = df.columns.str.strip()
        if "route_id" not in df.columns:
            return jsonify({"error": "No 'route_id' column found"}), 400
        df["route_id"] = df["route_id"].astype(str)
        if selected_routes:
            df = df[df["route_id"].isin(selected_routes)]
        period_cols = [c for c in df.columns if c != "route_id"]
        rowLabels = df["route_id"].tolist()
        matrix = []
        # Convert each cell: if missing, use None; otherwise cast to float.
        for _, row in df.iterrows():
            rowArr = [None if pd.isna(row[pc]) else float(row[pc]) for pc in period_cols]
            matrix.append(rowArr)
        return jsonify({
            "rowLabels": rowLabels,
            "colLabels": period_cols,
            "values": matrix,
            "title": "MPI Scores Heatmap"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500





@app.route("/visualize_eff_change", methods=["POST"])
def visualize_eff_change():
    data = request.get_json()
    excel_path = data.get("excelPath", "").strip()
    selected_routes = data.get("routes", [])
    if not excel_path or not os.path.exists(os.path.normpath(excel_path)):
        return jsonify({"error": "Efficiency Change file not found"}), 400
    try:
        excel_path = os.path.normpath(excel_path)
        df = pd.read_excel(excel_path)
        df.columns = df.columns.str.strip()
        if "route_id" not in df.columns:
            return jsonify({"error": "No 'route_id' column found"}), 400
        df["route_id"] = df["route_id"].astype(str)
        if selected_routes:
            df = df[df["route_id"].isin(selected_routes)]
        period_cols = [c for c in df.columns if c != "route_id"]
        rowLabels = df["route_id"].tolist()
        matrix = []
        for _, row in df.iterrows():
            rowArr = [None if pd.isna(row[pc]) else float(row[pc]) for pc in period_cols]
            matrix.append(rowArr)
        return jsonify({
            "rowLabels": rowLabels,
            "colLabels": period_cols,
            "values": matrix,
            "title": "Efficiency Change Heatmap"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.route("/visualize_tech_change", methods=["POST"])
def visualize_tech_change():
    data = request.get_json()
    excel_path = data.get("excelPath", "").strip()
    selected_routes = data.get("routes", [])
    if not excel_path or not os.path.exists(os.path.normpath(excel_path)):
        return jsonify({"error": "Technical Change file not found"}), 400
    try:
        excel_path = os.path.normpath(excel_path)
        df = pd.read_excel(excel_path)
        df.columns = df.columns.str.strip()
        if "route_id" not in df.columns:
            return jsonify({"error": "No 'route_id' column found"}), 400
        df["route_id"] = df["route_id"].astype(str)
        if selected_routes:
            df = df[df["route_id"].isin(selected_routes)]
        period_cols = [c for c in df.columns if c != "route_id"]
        rowLabels = df["route_id"].tolist()
        matrix = []
        for _, row in df.iterrows():
            rowArr = [None if pd.isna(row[pc]) else float(row[pc]) for pc in period_cols]
            matrix.append(rowArr)
        return jsonify({
            "rowLabels": rowLabels,
            "colLabels": period_cols,
            "values": matrix,
            "title": "Technological Change Heatmap"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500





@app.route("/contact")
def contact_page():
    return render_template("contact.html")

@app.route("/tools")
def tools_page():
    return render_template("tools.html")



# The Data engine codes: This part includes code realted to the different DEA models, different MPI modes and absolute efficiency score
def run_absolute_efficiency(path, weights_in, weights_out, results_dir):
    # Read the user’s Excel
    df = pd.read_excel(path)
    # Assume first two columns are DMU ID and DMU Name
    id_col, name_col = df.columns[0], df.columns[1]
    
    # Inputs are columns 3..(2+len(weights_in)), outputs next
    n_in  = len(weights_in)
    n_out = len(weights_out)
    inputs_df  = df.iloc[:, 2:2 + n_in].astype(float)
    outputs_df = df.iloc[:, 2 + n_in : 2 + n_in + n_out].astype(float)
    
    # Normalize by column maximum
    Xn = inputs_df  / inputs_df.max()
    Yn = outputs_df / outputs_df.max()
    
    # Compute score for each DMU
    # vectorized: numerator = Yn * weights_out, denominator = Xn * weights_in
    num = (Yn.values * weights_out).sum(axis=1)
    den = (Xn.values * weights_in).sum(axis=1)
    scores = [ (n/d if d else None) for n,d in zip(num, den) ]
    
    # Build results DataFrame
    res = pd.DataFrame({
      id_col: df[id_col],
      name_col: df[name_col],
      'Absolute_Efficiency': scores
    })
    
    # Save to results_dir
    base = os.path.splitext(os.path.basename(path))[0]
    out_name = f"{base}_Absolute_Efficiency_score.xlsx"
    out_path = os.path.join(results_dir, out_name)
    res.to_excel(out_path, index=False)




def run_ccr(path, orientation, n_in, n_out, results_dir):
    df = pd.read_excel(path)
    id_col, name_col = df.columns[0], df.columns[1]
    # slice input/output columns
    X = df.iloc[:, 2:2 + n_in].values
    Y = df.iloc[:, 2 + n_in:2 + n_in + n_out].values
    n, m = X.shape
    s = Y.shape[1]
    slack_inputs  = []   # will hold a list of length-m lists
    slack_outputs = []   # will hold a list of length-s lists
    base = os.path.splitext(os.path.basename(path))[0]
    out_name = f"{base}_DEA_CCR_{orientation}.xlsx"
    out_path = os.path.join(results_dir, out_name)

    eff_scores = []
    # Non-oriented needs slacks; epsilon for perturbation
    eps = 1e-6

    for o in range(n):
        if orientation == 'input':
            prob = LpProblem(f"CCR_Input_{o}", LpMinimize)
            theta = LpVariable('theta', lowBound=0)
            lamb = [LpVariable(f"l_{j}", lowBound=0) for j in range(n)]
            # objective
            prob += theta
            # input constraints
            for i in range(m):
                prob += lpSum(lamb[j] * X[j, i] for j in range(n)) <= theta * X[o, i]
            # output constraints
            for r in range(s):
                prob += lpSum(lamb[j] * Y[j, r] for j in range(n)) >= Y[o, r]

        elif orientation == 'output':
            prob = LpProblem(f"CCR_Output_{o}", LpMaximize)
            phi = LpVariable('phi', lowBound=0)
            lamb = [LpVariable(f"l_{j}", lowBound=0) for j in range(n)]
            # objective
            prob += phi
            # input constraints
            for i in range(m):
                prob += lpSum(lamb[j] * X[j, i] for j in range(n)) <= X[o, i]
            # output constraints
            for r in range(s):
                prob += lpSum(lamb[j] * Y[j, r] for j in range(n)) >= phi * Y[o, r]

        else:  # non-oriented
            prob = LpProblem(f"CCR_Non_{o}", LpMinimize)
            theta = LpVariable('theta', lowBound=0)
            lamb = [LpVariable(f"l_{j}", lowBound=0) for j in range(n)]
            s_minus = [LpVariable(f"s_minus_{i}", lowBound=0) for i in range(m)]
            s_plus  = [LpVariable(f"s_plus_{r}", lowBound=0) for r in range(s)]
            prob += theta - eps*(lpSum(s_minus) + lpSum(s_plus))
            for i in range(m):
                prob += lpSum(lamb[j]*X[j,i] for j in range(n)) + s_minus[i] == theta * X[o,i]
            for r in range(s):
                prob += lpSum(lamb[j]*Y[j,r] for j in range(n)) - s_plus[r] == Y[o,r]

        prob.solve(PULP_CBC_CMD(msg=False))

        # collect efficiency
        if orientation in ['input','non']:
            eff = theta.value()
        else:
            eff = phi.value()
        eff_scores.append(eff)
        
        # collect slack and surplus values
        if orientation == 'non':
            # capture the s_minus / s_plus you already created
            slack_inputs.append([v.value() for v in s_minus])
            slack_outputs.append([v.value() for v in s_plus])        

    # save results
    res_df = pd.DataFrame({
      id_col: df[id_col],
      name_col: df[name_col],
      'Efficiency': eff_scores
    })
    if orientation == 'non':
        # add each input slack as its own column
        for i in range(m):
            res_df[f'Slack_input_{i+1}'] = [row[i] for row in slack_inputs]
        # add each output surplus/slack
        for r in range(s):
            res_df[f'Slack_output_{r+1}'] = [row[r] for row in slack_outputs]


    res_df.to_excel(out_path, index=False)


def run_bcc(path, orientation, n_in, n_out, results_dir):
    import os
    import pandas as pd
    from pulp import LpProblem, LpMinimize, LpMaximize, LpVariable, lpSum, PULP_CBC_CMD

    # Read data
    df = pd.read_excel(path)
    id_col, name_col = df.columns[0], df.columns[1]
    X = df.iloc[:, 2:2 + n_in].values
    Y = df.iloc[:, 2 + n_in:2 + n_in + n_out].values
    n, m = X.shape
    s = Y.shape[1]
    base = os.path.splitext(os.path.basename(path))[0]
    out_name = f"{base}_DEA_BCC_{orientation}.xlsx"
    out_path = os.path.join(results_dir, out_name)

    eps = 1e-6
    eff_scores = []
    slack_inputs = []
    slack_outputs = []

    for o in range(n):
        # Build the LP
        if orientation == 'input':
            prob = LpProblem(f"BCC_Input_{o}", LpMinimize)
            lamb   = [LpVariable(f"l_{j}", lowBound=0) for j in range(n)]
            theta  = LpVariable("theta", lowBound=0)
            # CRS+VRS
            prob += lpSum(lamb) == 1
            prob += theta
            # constraints
            for i in range(m):
                prob += lpSum(lamb[j] * X[j,i] for j in range(n)) <= theta * X[o,i]
            for r in range(s):
                prob += lpSum(lamb[j] * Y[j,r] for j in range(n)) >= Y[o,r]

        elif orientation == 'output':
            prob = LpProblem(f"BCC_Output_{o}", LpMaximize)
            lamb = [LpVariable(f"l_{j}", lowBound=0) for j in range(n)]
            phi  = LpVariable("phi", lowBound=0)
            # VRS
            prob += lpSum(lamb) == 1
            prob += phi
            for i in range(m):
                prob += lpSum(lamb[j] * X[j,i] for j in range(n)) <= X[o,i]
            for r in range(s):
                prob += lpSum(lamb[j] * Y[j,r] for j in range(n)) >= phi * Y[o,r]

        else:  # non-oriented
            prob = LpProblem(f"BCC_Non_{o}", LpMinimize)
            lamb     = [LpVariable(f"l_{j}", lowBound=0) for j in range(n)]
            theta    = LpVariable("theta", lowBound=0)
            s_minus  = [LpVariable(f"s_minus_{i}", lowBound=0) for i in range(m)]
            s_plus   = [LpVariable(f"s_plus_{r}", lowBound=0) for r in range(s)]
            # Obj + VRS
            prob += theta - eps*(lpSum(s_minus) + lpSum(s_plus))
            prob += lpSum(lamb) == 1
            for i in range(m):
                prob += lpSum(lamb[j]*X[j,i] for j in range(n)) + s_minus[i] == theta * X[o,i]
            for r in range(s):
                prob += lpSum(lamb[j]*Y[j,r] for j in range(n)) - s_plus[r] == Y[o,r]

        prob.solve(PULP_CBC_CMD(msg=False))

        # Efficiency value
        if orientation in ('input','non'):
            eff = theta.value()
        else:
            eff = phi.value()
        eff_scores.append(eff)

        # Capture slacks for non-oriented
        if orientation == 'non':
            slack_inputs.append([v.value() for v in s_minus])
            slack_outputs.append([v.value() for v in s_plus])

    # Build results DataFrame
    res_df = pd.DataFrame({
      id_col: df[id_col],
      name_col: df[name_col],
      'Efficiency': eff_scores
    })

    # Append slack columns if non-oriented
    if orientation == 'non':
        for i in range(m):
            res_df[f"Slack_input_{i+1}"]  = [row[i] for row in slack_inputs]
        for r in range(s):
            res_df[f"Slack_output_{r+1}"] = [row[r] for row in slack_outputs]

    # Save
    res_df.to_excel(out_path, index=False)



def run_sbm(path, orientation, n_in, n_out, results_dir):

    # 1) read data
    df = pd.read_excel(path)
    DMUs = df.iloc[:,0].tolist()
    inputs = df.iloc[:, 2:2+n_in].values
    outputs = df.iloc[:, 2+n_in:2+n_in+n_out].values
    n, m = inputs.shape
    s = outputs.shape[1]

    eff_scores = []
    slacks_in = []
    slacks_out = []
    ref_sets = []

    for o in range(n):
        xi = inputs[o,:]
        yr = outputs[o,:]

        lamb = [LpVariable(f"λ_{j}", lowBound=0) for j in range(n)]
        s_minus = [LpVariable(f"s⁻_{i}", lowBound=0) for i in range(m)]
        s_plus = [LpVariable(f"s⁺_{r}", lowBound=0) for r in range(s)]
        theta = LpVariable("θ", lowBound=0)

        if orientation == 'input':
            prob = LpProblem(f"SBM_In_{o}", LpMinimize)
            # objective
            prob += theta - (1.0/m)*lpSum(s_minus[i]*(1.0/xi[i]) for i in range(m))
            # normalization
            prob += theta + (1.0/s)*lpSum(s_plus[r]*(1.0/yr[r]) for r in range(s)) == 1
            prob += lpSum(lamb) == 1
            for i in range(m):
                prob += lpSum(lamb[j]*inputs[j,i] for j in range(n)) + s_minus[i] == theta * xi[i]
            for r in range(s):
                prob += lpSum(lamb[j]*outputs[j,r] for j in range(n)) - s_plus[r] == theta * yr[r]

        elif orientation == 'output':
            prob = LpProblem(f"SBM_Out_{o}", LpMaximize)
            # objective
            prob += theta + (1.0/s)*lpSum(s_plus[r]*(1.0/yr[r]) for r in range(s))
            # normalization
            prob += theta - (1.0/m)*lpSum(s_minus[i]*(1.0/xi[i]) for i in range(m)) == 1
            prob += lpSum(lamb) == 1
            for i in range(m):
                prob += lpSum(lamb[j]*inputs[j,i] for j in range(n)) + s_minus[i] == theta * xi[i]
            for r in range(s):
                prob += lpSum(lamb[j]*outputs[j,r] for j in range(n)) - s_plus[r] == theta * yr[r]

        else:  # non-oriented
            inv_x = [0 if xi[i]==0 else 1/float(xi[i]) for i in range(m)]
            inv_y = [0 if yr[r]==0 else 1/float(yr[r]) for r in range(s)]
            prob = LpProblem(f"SBM_Non_{o}", LpMinimize)
            prob += theta - (1.0/m)*lpSum(s_minus[i]*inv_x[i] for i in range(m))
            prob += theta + (1.0/s)*lpSum(s_plus[r]*inv_y[r] for r in range(s)) == 1
            prob += lpSum(lamb) == 1
            for i in range(m):
                prob += theta*xi[i] - s_minus[i] == lpSum(lamb[j]*inputs[j,i] for j in range(n))
            for r in range(s):
                prob += theta*yr[r] + s_plus[r] == lpSum(lamb[j]*outputs[j,r] for j in range(n))

        prob.solve(PULP_CBC_CMD(msg=False))

        if LpStatus[prob.status] != "Optimal":
            eff_scores.append(None)
            ref_sets.append([])
            slacks_in.append([None]*m)
            slacks_out.append([None]*s)
        else:
            eff_val = theta.value()
            eff_scores.append(eff_val)
            ref_sets.append([DMUs[j] for j in range(n) if lamb[j].value() > 1e-6])
            slacks_in.append([s_minus[i].value() for i in range(m)])
            slacks_out.append([s_plus[r].value() for r in range(s)])

    ids = df.iloc[:,0].tolist()
    names = df.iloc[:,1].tolist()
    out = pd.DataFrame({
        'DMU': ids,
        'Route ID': names,
        'Efficiency': eff_scores,
        'Reference Set': ref_sets
    })

    for i in range(m):
        out[f"Slack_input_{i+1}"] = [row[i] for row in slacks_in]
    for r in range(s):
        out[f"Slack_output_{r+1}"] = [row[r] for row in slacks_out]

    fname = os.path.splitext(os.path.basename(path))[0]
    out.to_excel(os.path.join(results_dir, f"{fname}_DEA_SBM_{orientation}.xlsx"),
                 index=False)


# ——————————————————————————————————————————————————————————————————————
# 1) Pure‐solver versions of CCR/BCC/SBM that take two Excel paths:
#     path_ref  → builds the DEA frontier
#     path_eval → supplies the DMU inputs/outputs to score
#    and return a Python list of efficiencies in eval‐row order.
# ——————————————————————————————————————————————————————————————————————

# ——————————————————————————————————————————————————————————————————————
# Helper solvers for MPI (ASCII‐only variable names)
# ——————————————————————————————————————————————————————————————————————

def solve_ccr(path_ref, path_eval, orientation, n_in, n_out):
    import pandas as pd
    from pulp import LpProblem, LpMinimize, LpMaximize, LpVariable, lpSum, PULP_CBC_CMD
    df_ref  = pd.read_excel(path_ref)
    df_eval = pd.read_excel(path_eval)

    X_ref = df_ref.iloc[:, 2:2+n_in].values
    Y_ref = df_ref.iloc[:, 2+n_in:2+n_in+n_out].values
    X_ev  = df_eval.iloc[:, 2:2+n_in].values
    Y_ev  = df_eval.iloc[:, 2+n_in:2+n_in+n_out].values

    n, m = X_ref.shape
    s    = Y_ref.shape[1]
    eps  = 1e-6
    effs = []

    for o in range(len(df_eval)):
        x_o = X_ev[o]
        y_o = Y_ev[o]

        if orientation == 'input':
            prob = LpProblem(f"CCR_In_{o}", LpMinimize)
            theta = LpVariable('theta', lowBound=0)
            lamb  = [LpVariable(f"l_{j}", lowBound=0) for j in range(n)]
            prob += theta
            for i in range(m):
                prob += lpSum(lamb[j]*X_ref[j,i] for j in range(n)) <= theta * x_o[i]
            for r in range(s):
                prob += lpSum(lamb[j]*Y_ref[j,r] for j in range(n)) >= y_o[r]

        elif orientation == 'output':
            prob = LpProblem(f"CCR_Out_{o}", LpMaximize)
            phi    = LpVariable('phi', lowBound=0)
            lamb   = [LpVariable(f"l_{j}", lowBound=0) for j in range(n)]
            prob += phi
            for i in range(m):
                prob += lpSum(lamb[j]*X_ref[j,i] for j in range(n)) <= x_o[i]
            for r in range(s):
                prob += lpSum(lamb[j]*Y_ref[j,r] for j in range(n)) >= phi * y_o[r]

        else:  # non-oriented
            prob = LpProblem(f"CCR_Non_{o}", LpMinimize)
            theta   = LpVariable('theta', lowBound=0)
            lamb    = [LpVariable(f"l_{j}", lowBound=0) for j in range(n)]
            s_minus = [LpVariable(f"s_minus_{i}", lowBound=0) for i in range(m)]
            s_plus  = [LpVariable(f"s_plus_{r}", lowBound=0) for r in range(s)]
            prob += theta - eps*(lpSum(s_minus) + lpSum(s_plus))
            for i in range(m):
                prob += lpSum(lamb[j]*X_ref[j,i] for j in range(n)) + s_minus[i] == theta * x_o[i]
            for r in range(s):
                prob += lpSum(lamb[j]*Y_ref[j,r] for j in range(n)) - s_plus[r] == y_o[r]

        prob.solve(PULP_CBC_CMD(msg=False))
        if orientation in ('input','non'):
            effs.append(theta.value())
        else:
            effs.append(phi.value())

    return effs


def solve_bcc(path_ref, path_eval, orientation, n_in, n_out):
    import pandas as pd
    from pulp import LpProblem, LpMinimize, LpMaximize, LpVariable, lpSum, PULP_CBC_CMD

    df_ref  = pd.read_excel(path_ref)
    df_eval = pd.read_excel(path_eval)
    X_ref = df_ref.iloc[:, 2:2+n_in].values
    Y_ref = df_ref.iloc[:, 2+n_in:2+n_in+n_out].values
    X_ev  = df_eval.iloc[:, 2:2+n_in].values
    Y_ev  = df_eval.iloc[:, 2+n_in:2+n_in+n_out].values

    n, m = X_ref.shape
    s    = Y_ref.shape[1]
    eps  = 1e-6
    effs = []

    for o in range(len(df_eval)):
        x_o = X_ev[o]
        y_o = Y_ev[o]

        if orientation == 'input':
            prob = LpProblem(f"BCC_In_{o}", LpMinimize)
            lamb  = [LpVariable(f"l_{j}", lowBound=0) for j in range(n)]
            theta = LpVariable('theta', lowBound=0)
            prob += lpSum(lamb) == 1
            prob += theta
            for i in range(m):
                prob += lpSum(lamb[j]*X_ref[j,i] for j in range(n)) <= theta * x_o[i]
            for r in range(s):
                prob += lpSum(lamb[j]*Y_ref[j,r] for j in range(n)) >= y_o[r]

        elif orientation == 'output':
            prob = LpProblem(f"BCC_Out_{o}", LpMaximize)
            lamb = [LpVariable(f"l_{j}", lowBound=0) for j in range(n)]
            phi  = LpVariable('phi', lowBound=0)
            prob += lpSum(lamb) == 1
            prob += phi
            for i in range(m):
                prob += lpSum(lamb[j]*X_ref[j,i] for j in range(n)) <= x_o[i]
            for r in range(s):
                prob += lpSum(lamb[j]*Y_ref[j,r] for j in range(n)) >= phi * y_o[r]

        else:
            prob = LpProblem(f"BCC_Non_{o}", LpMinimize)
            lamb    = [LpVariable(f"l_{j}", lowBound=0) for j in range(n)]
            theta   = LpVariable('theta', lowBound=0)
            s_minus = [LpVariable(f"s_minus_{i}", lowBound=0) for i in range(m)]
            s_plus  = [LpVariable(f"s_plus_{r}", lowBound=0) for r in range(s)]
            prob += theta - eps*(lpSum(s_minus) + lpSum(s_plus))
            prob += lpSum(lamb) == 1
            for i in range(m):
                prob += lpSum(lamb[j]*X_ref[j,i] for j in range(n)) + s_minus[i] == theta * x_o[i]
            for r in range(s):
                prob += lpSum(lamb[j]*Y_ref[j,r] for j in range(n)) - s_plus[r] == y_o[r]

        prob.solve(PULP_CBC_CMD(msg=False))
        if orientation in ('input','non'):
            effs.append(theta.value())
        else:
            effs.append(phi.value())

    return effs


def solve_sbm(path_ref, path_eval, orientation, n_in, n_out):
    import pandas as pd
    from pulp import LpProblem, LpMinimize, LpMaximize, LpVariable, lpSum, PULP_CBC_CMD

    df_ref  = pd.read_excel(path_ref)
    df_eval = pd.read_excel(path_eval)
    X_ref = df_ref.iloc[:, 2:2+n_in].values
    Y_ref = df_ref.iloc[:, 2+n_in:2+n_in+n_out].values
    X_ev  = df_eval.iloc[:, 2:2+n_in].values
    Y_ev  = df_eval.iloc[:, 2+n_in:2+n_in+n_out].values

    n, m = X_ref.shape
    s    = Y_ref.shape[1]
    effs = []

    for o in range(len(df_eval)):
        x_o = X_ev[o]
        y_o = Y_ev[o]

        lamb    = [LpVariable(f"l_{j}", lowBound=0) for j in range(n)]
        s_minus = [LpVariable(f"s_minus_{i}", lowBound=0) for i in range(m)]
        s_plus  = [LpVariable(f"s_plus_{r}", lowBound=0) for r in range(s)]
        theta   = LpVariable('theta', lowBound=0)

        inv_x = [0 if x_o[i]==0 else 1.0/x_o[i] for i in range(m)]
        inv_y = [0 if y_o[r]==0 else 1.0/y_o[r] for r in range(s)]

        if orientation == 'input':
            prob = LpProblem(f"SBM_In_{o}", LpMinimize)
            prob += theta - (1.0/m)*lpSum(s_minus[i]*inv_x[i] for i in range(m))
            prob += theta + (1.0/s)*lpSum(s_plus[r]*inv_y[r] for r in range(s)) == 1
            prob += lpSum(lamb) == 1
            for i in range(m):
                prob += lpSum(lamb[j]*X_ref[j,i] for j in range(n)) + s_minus[i] == theta * x_o[i]
            for r in range(s):
                prob += lpSum(lamb[j]*Y_ref[j,r] for j in range(n)) - s_plus[r] == theta * y_o[r]

        elif orientation == 'output':
            prob = LpProblem(f"SBM_Out_{o}", LpMaximize)
            prob += theta + (1.0/s)*lpSum(s_plus[r]*inv_y[r] for r in range(s))
            prob += theta - (1.0/m)*lpSum(s_minus[i]*inv_x[i] for i in range(m)) == 1
            prob += lpSum(lamb) == 1
            for i in range(m):
                prob += lpSum(lamb[j]*X_ref[j,i] for j in range(n)) + s_minus[i] == theta * x_o[i]
            for r in range(s):
                prob += lpSum(lamb[j]*Y_ref[j,r] for j in range(n)) - s_plus[r] == theta * y_o[r]

        else:
            prob = LpProblem(f"SBM_Non_{o}", LpMinimize)
            prob += theta - (1.0/m)*lpSum(s_minus[i]*inv_x[i] for i in range(m))
            prob += theta + (1.0/s)*lpSum(s_plus[r]*inv_y[r] for r in range(s)) == 1
            prob += lpSum(lamb) == 1
            for i in range(m):
                prob += theta*x_o[i] - s_minus[i] == lpSum(lamb[j]*X_ref[j,i] for j in range(n))
            for r in range(s):
                prob += theta*y_o[r] + s_plus[r] == lpSum(lamb[j]*Y_ref[j,r] for j in range(n))

        prob.solve(PULP_CBC_CMD(msg=False))
        effs.append(theta.value())

    return effs



# ——————————————————————————————————————————————————————————————————————
# 2) The MPI runner
# ——————————————————————————————————————————————————————————————————————

def run_mpi(path_t, path_t1, model, orientation, n_in, n_out, results_dir):
    # pick the right solver
    solver = {
      'CCR': solve_ccr,
      'BCC': solve_bcc,
      'SBM': solve_sbm
    }[model]

    # 4 efficiencies
    E_t    = solver(path_t , path_t , orientation, n_in, n_out)
    E_t1   = solver(path_t1, path_t1, orientation, n_in, n_out)
    E_tt1  = solver(path_t , path_t1, orientation, n_in, n_out)
    E_t1t  = solver(path_t1, path_t , orientation, n_in, n_out)

    # compute EC, TC, MI
    EC = [ (e1/e0 if e0 and e1 else None) for e0,e1 in zip(E_t, E_t1) ]
    TC = [ ((a*b)**0.5 if a and b else None) for a,b in zip(E_tt1, E_t1t) ]
    MI = [ (ec*tc if ec and tc else None) for ec,tc in zip(EC, TC) ]

    df0 = pd.read_excel(path_t)
    ids   = df0.iloc[:,0]
    names = df0.iloc[:,1]

    out = pd.DataFrame({
      df0.columns[0]:     ids,
      df0.columns[1]:     names,
      'E_t':               E_t,
      'E_{t+1}':           E_t1,
      'E_t→t+1':           E_tt1,
      'E_{t+1}→t':         E_t1t,
      'EfficiencyChange':  EC,
      'TechChange':        TC,
      'MalmquistIndex':    MI
    })

    b0  = os.path.splitext(os.path.basename(path_t ))[0]
    b1  = os.path.splitext(os.path.basename(path_t1))[0]
    fname = f"{b0}_{b1}_{model}_{orientation}_MPI.xlsx"
    out.to_excel(os.path.join(results_dir, fname), index=False)




@app.route("/data")
def data():
    return render_template("data.html")



@app.route('/compute_data_engine', methods=['POST'])
def compute_data_engine():
    calc = request.form.get('calc_type')
    results_dir = os.path.join(app.root_path, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # 1) Absolute efficiency
    if calc == 'absolute':
        # Gather the single/multiple file paths
        raw = request.form.getlist('period_paths')
        paths = [p.strip() for p in raw if p.strip()]
        if not paths:
            return "Error: please provide at least one valid data‐file path.", 400
        # Gather weights
        w_in  = [float(w) for w in request.form.getlist('input_weights')]
        w_out = [float(w) for w in request.form.getlist('output_weights')]

        # Run and save one file per period
        for path in paths:
            run_absolute_efficiency(path, w_in, w_out, results_dir)

        return f"Absolute efficiency results saved in: {results_dir}"

    # 2) Relative efficiency
    elif calc == 'relative':
        model       = request.form.get('dea_model')
        orientation = request.form.get('orientation')
        n_in        = int(request.form.get('var_num_inputs'))
        n_out       = int(request.form.get('var_num_outputs'))
        raw = request.form.getlist('period_paths')
        paths = [p.strip() for p in raw if p.strip()]
        if not paths:
            return "Error: please provide at least one valid data‐file path.", 400

        for path in paths:
            if model == 'CCR':
                run_ccr(path, orientation, n_in, n_out, results_dir)
            elif model == 'BCC':
                run_bcc(path, orientation, n_in, n_out, results_dir)
            else:
                run_sbm(path, orientation, n_in, n_out, results_dir)

        return f"{model} ({orientation}) results saved in: {results_dir}"


    # 3) MPI 
    elif calc == 'mpi':
        # gather form inputs
        model       = request.form['dea_model']
        orientation = request.form['orientation']
        n_in        = int(request.form['var_num_inputs'])
        n_out       = int(request.form['var_num_outputs'])
        pths        = request.form.getlist('mpi_paths')
        if len(pths) != 2:
            return "Error: need exactly two period paths.", 400

        # call our new runner
        run_mpi(pths[0], pths[1], model, orientation, n_in, n_out, results_dir)
        return f"MPI ({model},{orientation}) results saved in: {results_dir}"


    # 4) Unknown
    else:
        return "Unknown calculation type", 400



###############################################################################
#  Access Tool
###############################################################################


# —————————————————————————————————————————————————————————————
# 1) Helper: unzip shapefile
# —————————————————————————————————————————————————————————————
def _unzip_shapefile(zip_path: str, dest_folder: str, subfolder_name: str) -> str:
    """
    Unzip a .zip containing a shapefile into a subfolder, return that subfolder path.
    """
    out_folder = os.path.join(dest_folder, subfolder_name)
    os.makedirs(out_folder, exist_ok=True)

    # <<< ADD DEBUG: confirm zip_path exists and is a file
    print("DEBUG [_unzip_shapefile]: zip_path =", zip_path, "exists?", os.path.exists(zip_path), "isfile?", os.path.isfile(zip_path), flush=True)



    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(out_folder)

    # <<< ADD DEBUG: confirm extraction folder contents
    print("DEBUG [_unzip_shapefile]: out_folder =", out_folder, "contains:", os.listdir(out_folder), flush=True)


    return out_folder

# —————————————————————————————————————————————————————————————
# 2) Helper: read TAZ shapefile (filter pd == 36, reproject, add ID)
# —————————————————————————————————————————————————————————————
def _read_taz_shp(shp_path: str) -> gpd.GeoDataFrame:
    """
    Read the TAZ shapefile, filter to pd==36 (Mississauga),
    ensure CRS=EPSG:4326, return GeoDataFrame with integer 'id' column.
    """
    # <<< ADD DEBUG: confirm shp_path exists
    print("DEBUG [_read_taz_shp]: shp_path =", shp_path, "exists?", os.path.exists(shp_path), flush=True)

    # Some zipped shapefiles may be missing .shx → enable restore:
    with Env(SHAPE_RESTORE_SHX="YES"):
        gdf = gpd.read_file(shp_path)

    # <<< ADD DEBUG: after reading, show columns & CRS
    print("DEBUG [_read_taz_shp]: initial gdf columns:", gdf.columns.tolist(), "CRS:", gdf.crs, flush=True)

    if 'pd' not in gdf.columns:
        raise Exception("TAZ shapefile is missing a 'pd' column.")
    miss = gdf[gdf['pd'] == 36].copy()
    # ← **Make sure geometry is still the active geometry column:**
    miss = miss.set_geometry(miss.geometry.name)

    print("DEBUG [_read_taz_shp]: filtered miss rows:", len(miss), flush=True)

    if miss.crs is None:
        miss.set_crs(epsg=4326, inplace=True)
    else:
        miss = miss.to_crs(epsg=4326)

    print("DEBUG [_read_taz_shp]: miss CRS after reproject:", miss.crs, flush=True)

    miss = miss.reset_index(drop=True)
    miss['id'] = miss.index  # assign TAZ‐ID as index


    print("DEBUG [_read_taz_shp]: 'id' column assigned; sample ids:", miss['id'].head().tolist(), flush=True)

    return miss

# —————————————————————————————————————————————————————————————
# 3) Helper: extract grocery points from geojson
# —————————————————————————————————————————————————————————————
def _extract_points_from_geojson(geojson_path: str, bbox: tuple) -> gpd.GeoDataFrame:
    """
    Load a GeoJSON (e.g. OSM export), filter to shops = ['supermarket','grocery'],
    extract lon/lat from geometry, filter to bbox, return a GeoDataFrame with 'id' column.
    """

    # <<< ADD DEBUG: confirm geojson_path exists
    print("DEBUG [_extract_points_from_geojson]: geojson_path =", geojson_path, "exists?", os.path.exists(geojson_path), flush=True)

    raw = gpd.read_file(geojson_path)
    print("DEBUG [_extract_points_from_geojson]: raw columns:", raw.columns.tolist(), "CRS:", raw.crs, flush=True)

    if 'shop' not in raw.columns:
        raise Exception("GeoJSON does not contain a 'shop' field.")
    pts = raw[ raw['shop'].isin(['supermarket','grocery']) ].copy()
    pts = pts.set_geometry('geometry')
    print("DEBUG [_extract_points_from_geojson]: pts rows after filter:", len(pts), flush=True)


    if pts.empty:
        raise Exception("Uploaded GeoJSON has zero 'supermarket' or 'grocery' points.")

    # Extract lon/lat directly from the Point geometry (no .centroid needed)
    pts['lon'] = pts.geometry.x
    pts['lat'] = pts.geometry.y

    # <<< ADD DEBUG: show bounding‐box values
    print("DEBUG [_extract_points_from_geojson]: bbox =", bbox, flush=True)

    # filter to bounding‐box (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = bbox
    pts = pts[
       (pts['lon'] >= minx) & (pts['lon'] <= maxx) &
       (pts['lat'] >= miny) & (pts['lat'] <= maxy)
    ].copy()
    print("DEBUG [_extract_points_from_geojson]: pts rows after bbox filter:", len(pts), flush=True)

    if pts.empty:
        raise Exception("After bounding‐box filter, no grocery points remain inside the city boundary.")

    pts = pts.reset_index(drop=True)
    pts['id'] = pts.index

    if pts.crs is None:
        pts.set_crs(epsg=4326, inplace=True)
    else:
        pts = pts.to_crs(epsg=4326)

    # Ensure 'name' column exists (some GeoJSONs don’t have it)
    if 'name' not in pts.columns:
        pts['name'] = ""
    print("DEBUG [_extract_points_from_geojson]: returned columns:", pts[['id','lon','lat','name']].head(), flush=True)
    return pts[['id','geometry','lon','lat','name']]

# —————————————————————————————————————————————————————————————
# 4) Helper: build a brand-new R5 network each time
# —————————————————————————————————————————————————————————————
def _build_transport_network(osm_pbf_path: str,
                             gtfs_zip_path: str) -> r5py.TransportNetwork:
    """
    Build and return an r5py TransportNetwork from OSM PBF + GTFS ZIP,
    forcing R5 to store its Lucene index inside `data_directory`,
    and disabling any existing cache.
    """
    # <<< ADD DEBUG: check that both inputs exist
    print("DEBUG [_build_transport_network]: osm_pbf_path =", osm_pbf_path, "exists?", os.path.exists(osm_pbf_path), flush=True)
    print("DEBUG [_build_transport_network]: gtfs_zip_path =", gtfs_zip_path, "exists?", os.path.exists(gtfs_zip_path), flush=True)

    tn = r5py.TransportNetwork(osm_pbf_path, [gtfs_zip_path])
    # <<< ADD DEBUG: after building, confirm the object
    print("DEBUG [_build_transport_network]: TransportNetwork created:", tn, flush=True)

    return tn


# —————————————————————————————————————————————————————————————
# 5) Helper: compute single departure "nearest" matrix (FINAL FIX)
# —————————————————————————————————————————————————————————————
def _find_nearest_matrix(
    tn: r5py.TransportNetwork,
    origins_gdf: gpd.GeoDataFrame,
    destinations_gdf: gpd.GeoDataFrame,
    departure: datetime.datetime,
    walk_speed: float,
    max_walk_s: int
) -> pd.DataFrame:
    """
    Compute a single‐departure travel time matrix at `departure`, then pick
    for each origin ID the destination ID with minimum travel_time.
    Returns a DataFrame with columns ['from_id','to_id'].
    """
    matrix = r5py.TravelTimeMatrix(
        transport_network=tn,
        origins=origins_gdf,
        destinations=destinations_gdf,
        transport_modes=[r5py.TransportMode.TRANSIT, r5py.TransportMode.WALK],
        departure=departure,
        speed_walking=walk_speed,
        max_time_walking=datetime.timedelta(seconds=max_walk_s)
    )

    # Convert the r5py matrix object into a true, clean Pandas DataFrame.
    matrix = pd.DataFrame(matrix)

    print("DEBUG [_find_nearest_matrix]: raw travel‐time matrix rows:", len(matrix), flush=True)

    matrix = matrix.dropna(subset=['from_id','travel_time'])
    if matrix.empty:
        return pd.DataFrame(columns=['from_id','to_id'])

    nearest = matrix.loc[
        matrix.groupby('from_id')['travel_time'].idxmin()
    ][['from_id','to_id']]
    
    print("DEBUG [_find_nearest_matrix]: nearest shape:", nearest.shape, flush=True)
    return nearest.reset_index(drop=True)


# —————————————————————————————————————————————————————————————
# 6) Helper: average travel times over interval (FINAL FIX)
# —————————————————————————————————————————————————————————————
def _average_travel_times_over_interval(
    tn: r5py.TransportNetwork,
    origins_gdf: gpd.GeoDataFrame,
    destinations_gdf: gpd.GeoDataFrame,
    nearest_df: pd.DataFrame,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    walk_speed: float,
    max_walk_s: int,
    out_csv_path: str
) -> None:
    """
    For each minute between start_time and end_time (inclusive),
    compute travel times, filter only (from_id,to_id) in nearest_df, collect them,
    then average per from_id, and write to out_csv_path.
    """
    interval = datetime.timedelta(minutes=5)
    collected = []
    current = start_time

    while current <= end_time:
        matrix = r5py.TravelTimeMatrix(
            transport_network=tn,
            origins=origins_gdf,
            destinations=destinations_gdf,
            transport_modes=[r5py.TransportMode.TRANSIT, r5py.TransportMode.WALK],
            departure=current,
            speed_walking=walk_speed,
            max_time_walking=datetime.timedelta(seconds=max_walk_s)
        )
        
        # Convert the r5py matrix object into a true, clean Pandas DataFrame.
        matrix = pd.DataFrame(matrix)
        
        matrix = matrix.dropna(subset=['from_id','to_id','travel_time'])

        if not matrix.empty and not nearest_df.empty:
            merged = matrix.merge(nearest_df, on=['from_id','to_id'], how='inner')
            collected.append(merged[['from_id','travel_time']])

        current += interval

    if not collected:
        avg_df = pd.DataFrame(columns=['from_id','avg_travel_time'])
    else:
        all_times = pd.concat(collected, ignore_index=True)
        avg_df = all_times.groupby('from_id')['travel_time'].mean().reset_index()
        avg_df.rename(columns={'travel_time':'avg_travel_time'}, inplace=True)

    avg_df.to_csv(out_csv_path, index=False)
    print("DEBUG [_average_travel_times_over_interval]: wrote CSV to", out_csv_path, "exists now?", os.path.exists(out_csv_path), flush=True)


# —————————————————————————————————————————————————————————————
# 7a) New Helper For Hospitals and Schools
# —————————————————————————————————————————————————————————————

def _extract_points_from_shapefile(zip_path: str, bbox: tuple) -> gpd.GeoDataFrame:
    """
    Unzip a ZIP of a point-shapefile (e.g. hospitals), load its .shp,
    reproject to EPSG:4326, clip to the city boundary bbox, ensure an 'id' column, and return it.
    """
    # 1. Unzip into a service_shp subfolder
    service_folder = _unzip_shapefile(zip_path, os.path.dirname(zip_path), "service_shp")

    # 2. Find & read the .shp inside
    shp_file = next(
        (os.path.join(service_folder, f)
         for f in os.listdir(service_folder)
         if f.lower().endswith(".shp")),
        None
    )
    if not shp_file:
        raise Exception("No .shp file found inside uploaded service ZIP.")
    pts = gpd.read_file(shp_file)

    # 3. Reproject *first* to WGS84 lon/lat so we can clip by your boundary bbox
    if pts.crs:
        pts = pts.to_crs(epsg=4326)
    else:
        pts.set_crs(epsg=4326, inplace=True)

    # 4. If features aren’t already points, use centroids
    if not all(pts.geometry.geom_type == "Point"):
        pts["geometry"] = pts.geometry.centroid

    # 5. Clip to the city boundary bbox (in EPSG:4326)
    minx, miny, maxx, maxy = bbox
    pts = pts.cx[minx:maxx, miny:maxy].copy()
    if pts.empty:
        raise Exception("No hospital points fall inside the city boundary.")

    # 6. Build the fields we need
    pts = pts.reset_index(drop=True)
    pts["id"] = pts.index
    pts["lon"] = pts.geometry.x
    pts["lat"] = pts.geometry.y

    return pts[["id","geometry","lon","lat"]]


# —————————————————————————————————————————————————————————————
# 7b) New Helper For Jobs
# —————————————————————————————————————————————————————————————

def _extract_jobs_destinations(zip_path: str, taz_gdf: gpd.GeoDataFrame, bbox: tuple) -> gpd.GeoDataFrame:
    """
    1. Unzip a ZIP of the business-directory shapefile.
    2. Read it, reproject to EPSG:4326.
    3. Spatial-join to count businesses within each TAZ.
    4. Build a GeoDataFrame of TAZ centroids with a 'business_count' column.
    """
    # unzip
    svc = _unzip_shapefile(zip_path, os.path.dirname(zip_path), "service_shp")
    shp = next((os.path.join(svc,f) for f in os.listdir(svc) if f.lower().endswith(".shp")), None)
    if not shp:
        raise Exception("No .shp in jobs ZIP.")
    jobs = gpd.read_file(shp)
    # reproject to match TAZ and bbox
    if jobs.crs:
        jobs = jobs.to_crs(epsg=4326)
    else:
        jobs.set_crs(epsg=4326, inplace=True)
    # clip to city bbox
    minx,miny,maxx,maxy = bbox
    jobs = jobs.cx[minx:maxx, miny:maxy].copy()
    if jobs.empty:
        raise Exception("No job locations inside the city boundary.")
    # spatial‐join to count per TAZ
    join = gpd.sjoin(jobs, taz_gdf[['id','geometry']], how='inner', predicate='within')
    counts = join.groupby('id').size().reset_index(name='business_count')
    # merge back to TAZ
    merged = taz_gdf[['id','geometry']].merge(counts, on='id', how='left')
    merged['business_count'] = merged['business_count'].fillna(0)
    # build centroids
    merged['geometry'] = merged.geometry.centroid
    return merged[['id','geometry','business_count']]


# —————————————————————————————————————————————————————————————
# 7b) New Helper For Average number of reachable Jobs
# —————————————————————————————————————————————————————————————

def _calculate_average_jobs_reachable(
    tn: r5py.TransportNetwork,
    origins: gpd.GeoDataFrame,
    destinations: gpd.GeoDataFrame,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    walk_speed: float,
    max_walk_s: int,
    reach_time_s: int,
    out_csv: str
):
    """
    For each minute in [start_time,end_time]:
      - compute travel-time matrix,
      - filter travel_time <= reach_time_s,
      - merge with destinations['business_count'],
      - sum per origin,
    then average those sums and write CSV as
    ['from_id','avg_business_count'].
    """
    interval = datetime.timedelta(minutes=1)
    records = []
    t = start_time
    while t <= end_time:
        m = r5py.TravelTimeMatrix(
            transport_network=tn,
            origins=origins,
            destinations=destinations,
            transport_modes=[r5py.TransportMode.TRANSIT, r5py.TransportMode.WALK],
            departure=t,
            speed_walking=walk_speed,
            max_time_walking=datetime.timedelta(seconds=max_walk_s)
        )
        df = pd.DataFrame(m).dropna(subset=['from_id','to_id','travel_time'])
        df = df[df['travel_time'] <= reach_time_s]
        if not df.empty:
            df = df.merge(destinations[['id','business_count']],
                          left_on='to_id', right_on='id', how='left')
            sums = df.groupby('from_id')['business_count'].sum().reset_index()
            records.append(sums)
        t += interval

    if records:
        all_times = pd.concat(records, ignore_index=True)
        avg = all_times.groupby('from_id')['business_count'].mean().reset_index()
    else:
        avg = pd.DataFrame(columns=['from_id','business_count'])
    avg.to_csv(out_csv, index=False)



# —————————————————————————————————————————————————————————————
# 8) The /access route
# —————————————————————————————————————————————————————————————
@app.route("/access", methods=["GET", "POST"])
def access():
    """
    GET → render blank map + form (access.html).
    POST → process uploads, compute travel times, merge to TAZ, build choropleth.
    """
    if request.method == "GET":
        m = folium.Map(location=[43.6532, -79.3832], zoom_start=11, tiles="OpenStreetMap")
        folium.plugins.Fullscreen(position='topright').add_to(m)
        map_html = m._repr_html_()
        return render_template(
            "access.html",
            map_html=map_html,
            prev_access_type="Grocery",
            prev_basemap="OpenStreetMap"
        )

    # ——— Handle POST ———
    session_id = str(uuid.uuid4())
    base_upload = os.path.join(app.root_path, "uploads", session_id)
    base_output = os.path.join(app.root_path, "outputs", session_id)
    os.makedirs(base_upload, exist_ok=True)
    os.makedirs(base_output, exist_ok=True)
    
    r5_scratch_folder = None
    old_cwd = os.getcwd()

    try:
        # 1) Read form fields and save uploaded files
        access_type = request.form.get("access_type", "").strip()
        walk_speed  = float(request.form.get("walk_speed") or 1.39)
        max_walk_s  = int(request.form.get("max_walk_s") or 360)
        base_map    = request.form.get("base_map") or "OpenStreetMap"
        tr_count    = int(request.form.get("TR_count") or 0)


        taz_zip_path = os.path.join(base_upload, "taz_shp.zip")
        boundary_zip_path = os.path.join(base_upload, "boundary_shp.zip")
        service_path = os.path.join(base_upload, "service.geojson")
        gtfs_zip_path = os.path.join(base_upload, "gtfs.zip")
        osm_pbf_path = os.path.join(base_upload, "osm.pbf")

        request.files["taz_shp_zip"].save(taz_zip_path)
        request.files["boundary_shp_zip"].save(boundary_zip_path)
        request.files["service_file"].save(service_path)
        request.files["gtfs_zip"].save(gtfs_zip_path)
        request.files["osm_pbf"].save(osm_pbf_path)
        
        # 2) Unzip shapefiles and prepare all GeoDataFrames (non-R5 work)
        taz_folder = _unzip_shapefile(taz_zip_path, base_upload, "taz_shp")
        boundary_folder = _unzip_shapefile(boundary_zip_path, base_upload, "boundary_shp")
        
        taz_shp_file = next((os.path.join(taz_folder, fn) for fn in os.listdir(taz_folder) if fn.lower().endswith(".shp")), None)
        if not taz_shp_file: raise Exception("No .shp file found inside uploaded TAZ zip.")
        taz_gdf = _read_taz_shp(taz_shp_file)

        taz_for_centroids_utm = taz_gdf.to_crs(epsg=32617)
        centroids_utm = taz_for_centroids_utm.geometry.centroid
        origins_gdf = gpd.GeoDataFrame(data={'id': taz_gdf['id']}, geometry=centroids_utm, crs="EPSG:32617").to_crs(epsg=4326)
        if origins_gdf.geometry.name != 'geometry': origins_gdf.rename_geometry('geometry', inplace=True)
        
        boundary_shp_file = next((os.path.join(boundary_folder, fn) for fn in os.listdir(boundary_folder) if fn.lower().endswith(".shp")), None)
        if not boundary_shp_file: raise Exception("No .shp file found inside uploaded Boundary zip.")
        boundary_gdf = gpd.read_file(boundary_shp_file)
        if boundary_gdf.crs is not None: boundary_gdf = boundary_gdf.to_crs(epsg=4326)
        else: boundary_gdf.set_crs(epsg=4326, inplace=True)

        bbox = tuple(boundary_gdf.total_bounds)
        # build the correct destinations layer based on service type
        if access_type == "Grocery":
            destinations_gdf = _extract_points_from_geojson(service_path, bbox)
        elif access_type in ("Hospital", "Schools"):
            # both hospital and school shapefiles come in as a ZIP of a point-shapefile
            destinations_gdf = _extract_points_from_shapefile(service_path, bbox)
        elif access_type == "Jobs":
            # business-directory ZIP → TAZ centroids + business_count
            destinations_gdf = _extract_jobs_destinations(service_path, taz_gdf, bbox)
        else:
            raise Exception(f"{access_type} access not implemented yet.")

        # 3) Perform ALL R5-related computations inside a managed CWD block
        r5_scratch_folder = tempfile.mkdtemp(prefix="r5_scratch_")
        tn = None
        csv_files = [] 

        try:
            os.chdir(r5_scratch_folder)
            tn = _build_transport_network(osm_pbf_path, gtfs_zip_path)
            
            intervals = []
            for i in range(1, tr_count + 1):
                start_str, end_str, label = request.form.get(f"start_{i}"), request.form.get(f"end_{i}"), request.form.get(f"label_{i}", "").strip()
                if not all((start_str, end_str, label)): continue
                if not label.lower().endswith(".csv"): label += ".csv"
                intervals.append((datetime.datetime.fromisoformat(start_str), datetime.datetime.fromisoformat(end_str), label))
            
            if not intervals: raise Exception("At least one valid time-range interval must be provided.")

            for (dt_start, dt_end, label) in intervals:
                out_csv = os.path.join(base_output, label)
                if access_type == "Jobs":
                    # count jobs reachable within 45 min (2700s)
                    _calculate_average_jobs_reachable(
                        tn, origins_gdf, destinations_gdf,
                        dt_start, dt_end,
                        walk_speed, max_walk_s,
                        2700,  # 45 minutes - You can play with this number to chnage the treshold
                        out_csv
                    )
                else:
                    nearest_df = _find_nearest_matrix(
                        tn, origins_gdf, destinations_gdf,
                        dt_start, walk_speed, max_walk_s
                    )
                    _average_travel_times_over_interval(
                        tn, origins_gdf, destinations_gdf, nearest_df,
                        dt_start, dt_end, walk_speed, max_walk_s, out_csv
                    )
                csv_files.append(label)


        finally:
            os.chdir(old_cwd)
            if tn is not None:
                del tn
                gc.collect()
            if r5_scratch_folder and os.path.exists(r5_scratch_folder):
                shutil.rmtree(r5_scratch_folder, ignore_errors=True)

        # 4) Now, with all computations done, build the map
        if not csv_files:
            raise Exception("Calculation completed, but no output CSV files were generated.")
        
        first_label = csv_files[0]
        first_csv_path = os.path.join(base_output, first_label)
        avg_df = pd.read_csv(first_csv_path)
        
        merged = taz_gdf.merge(avg_df, left_on="id", right_on="from_id", how="left")
        center_point = merged.geometry.centroid.unary_union.centroid
        center_lat, center_lon = center_point.y, center_point.x


        m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles=base_map)
        folium.plugins.Fullscreen(position='topright').add_to(m)
        folium.GeoJson(boundary_gdf.to_json(), style_function=lambda feat: {"color":"black", "weight":1, "fillOpacity":0}).add_to(m)

        if access_type == "Jobs":
            fld = "business_count"
            display = "Avg Jobs Reachable"
        else:
            fld = "avg_travel_time"
            display = "Avg Travel Time (s)"

        vals = merged[fld].dropna()
        if not vals.empty:
            vmin, vmax = float(vals.min()), float(vals.max())
            cmap = (cm.LinearColormap(
                ["red","orange","yellow","green"] if access_type=="Jobs"
                else ["green","yellow","orange","red"],
                vmin=vmin, vmax=vmax
            ))
            cmap.caption = f"{display} ({first_label.split('.')[0]})"

            def style_function(feature):
                val = feature["properties"].get(fld)
                if val is None:
                    return {"fillColor": "#810303", "color":"black", "weight":1, "fillOpacity":0.9}
                else:
                    return {
                  "fillColor": cmap(val),
                  "color": "black",
                  "weight": 1,
                  "fillOpacity": 0.9
                }            

            folium.GeoJson(
                data=merged.to_json(),
                style_function=style_function,
                highlight_function=lambda feat: {"weight":2, "color":"blue"},
                tooltip=folium.GeoJsonTooltip(
                    fields=["id", fld],
                    aliases=["TAZ ID", display],
                    localize=True,
                    sticky=True
                )
            ).add_to(m)
            cmap.add_to(m)
        else:
            folium.GeoJson(data=merged.to_json(), style_function=lambda feat: {"fillColor":"#808080", "color":"black", "weight":1, "fillOpacity":0.3}).add_to(m)
            
        map_html = m._repr_html_()
        download_links = [url_for('download_access_file', session_id=session_id, filename=fname) for fname in csv_files]
        
        return render_template(
            "access.html",
            map_html=map_html,
            download_links=download_links,
            prev_access_type=access_type,
            prev_basemap=base_map
        )

    except Exception as e:
        print(f"ERROR [/access POST outer except]: {type(e).__name__} - {e}", flush=True)
        import traceback
        traceback.print_exc()

        if os.getcwd() != old_cwd: os.chdir(old_cwd)
        if 'r5_scratch_folder' in locals() and r5_scratch_folder and os.path.exists(r5_scratch_folder):
            shutil.rmtree(r5_scratch_folder, ignore_errors=True)
        shutil.rmtree(base_upload, ignore_errors=True)
        shutil.rmtree(base_output, ignore_errors=True)

        err_map = folium.Map(location=[43.6532, -79.3832], zoom_start=11, tiles="OpenStreetMap")
        folium.plugins.Fullscreen(position='topright').add_to(err_map)
        return render_template(
            "access.html",
            map_html=err_map._repr_html_(),
            prev_access_type=request.form.get("access_type", "Grocery"),
            prev_basemap=request.form.get("base_map", "OpenStreetMap"),
            error_message=str(e)
        )

# —————————————————————————————————————————————————————————————
# 9) Download endpoint
# —————————————————————————————————————————————————————————————
@app.route("/download_access/<session_id>/<filename>")
def download_access_file(session_id, filename):
    folder = os.path.join(app.root_path, "outputs", session_id)
    path = os.path.join(folder, filename)
    
    print("DEBUG [/download_access]: Serving file", path, flush=True)

    return send_file(
        path,
        as_attachment=True,
        download_name=filename  
    )

# —————————————————————————————————————————————————————————————
# 10) Read GTFS dates
# —————————————————————————————————————————————————————————————
@app.route("/access/read_gtfs_dates", methods=["POST"])
def access_read_gtfs_dates():
    if 'gtfs_zip' not in request.files:
        return jsonify({"error": "No GTFS ZIP provided"}), 400

    temp_id = str(uuid.uuid4())
    temp_folder = os.path.join(app.root_path, "uploads", temp_id)
    os.makedirs(temp_folder, exist_ok=True)
    print("DEBUG [/access/read_gtfs_dates]: temp_folder =", temp_folder, flush=True)

    try:
        gz = request.files['gtfs_zip']
        zip_path = os.path.join(temp_folder, "temp_gtfs.zip")
        print("DEBUG [/access/read_gtfs_dates]: saving GTFS zip to", zip_path, flush=True)
        gz.save(zip_path)

        with zipfile.ZipFile(zip_path, 'r') as z:
            names = z.namelist()
            print("DEBUG [/access/read_gtfs_dates]: names in zip =", names, flush=True)
            candidate = None
            for name in names:
                lower = name.lower()
                if lower.endswith("calendar_dates.txt"):
                    candidate = name
                    break
            if candidate is None:
                for name in names:
                    if name.lower().endswith("calendar.txt"):
                        candidate = name
                        break
            if candidate is None:
                return jsonify({"error": "Neither 'calendar_dates.txt' nor 'calendar.txt' found in GTFS zip."}), 400

            z.extract(candidate, temp_folder)
            print("DEBUG [/access/read_gtfs_dates]: extracted candidate =", candidate, flush=True)

        cal_path = os.path.join(temp_folder, candidate)
        print("DEBUG [/access/read_gtfs_dates]: cal_path =", cal_path, "exists?", os.path.exists(cal_path), flush=True)
        
        cal_df = pd.read_csv(cal_path, dtype={'date': str})
        print("DEBUG [/access/read_gtfs_dates]: cal_df head:", cal_df.head(), flush=True)

        if 'date' not in cal_df.columns:
            return jsonify({"error": "The GTFS calendar file does not contain a 'date' column."}), 400

        start_date = cal_df['date'].min()
        end_date   = cal_df['date'].max()
        print("DEBUG [/access/read_gtfs_dates]: start_date =", start_date, "end_date =", end_date, flush=True)
        

        return jsonify({"start_date": start_date, "end_date": end_date})
    except Exception as e:
        print("ERROR [/access/read_gtfs_dates]:", type(e).__name__, "-", e, flush=True)
        return jsonify({"error": str(e)}), 500
    finally:
        print("DEBUG [/access/read_gtfs_dates]: cleaning temp_folder", temp_folder, flush=True)
        shutil.rmtree(temp_folder, ignore_errors=True)

@app.route("/help")
def help_page():
    # This will render a new template that embeds your PDF.
    return render_template("help.html")



if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
