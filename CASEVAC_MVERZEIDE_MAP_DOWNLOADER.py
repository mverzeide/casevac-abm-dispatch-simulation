"""
CASEVAC Map Downloader

This script downloads and prepares a road network from OpenStreetMap using OSMnx.
The generated GraphML file is used as the road-network input for the CASEVAC
simulation scripts.

Default output:
    gulpen_polygon_simplified.graphml

The simulation scripts expect this file to be available in the same working
directory as the Python scripts.
"""

# region Libraries
# This block imports the required geospatial and plotting libraries.
import math

import matplotlib.pyplot as plt
import osmnx as ox
from shapely.geometry import Polygon
# endregion


# region OSMnx Settings
# This block configures OSMnx and Overpass API settings for downloading map data.
ox.settings.overpass_url = "https://overpass-api.de/api/interpreter"
ox.settings.requests_timeout = 300
ox.settings.use_cache = True
ox.settings.log_console = True
ox.settings.max_query_area_size = 5e9
# endregion


# region Region Configuration
# This block defines the available study areas and their bounding-box dimensions.
CITY = "Gulpen"
# Available options:
# "Gulpen", "Bergen", "Valkenburg", "Lapland", "Almelo",
# "Houten", "Sneek", "Varėna", "Nicosia"

REGION_CONFIGS = {
    "Gulpen": {
        "center_lat": 50.8167,
        "center_lon": 5.8833,
        "width_km": 12,
        "height_km": 12,
        "graph_file": "gulpen_polygon_simplified.graphml",
    },
    "Bergen": {
        "center_lat": 60.39299,
        "center_lon": 5.32415,
        "width_km": 15,
        "height_km": 15,
        "graph_file": "bergen_polygon_simplified.graphml",
    },
    "Valkenburg": {
        "center_lat": 50.8610,
        "center_lon": 5.8285,
        "width_km": 8,
        "height_km": 8,
        "graph_file": "valkenburg_polygon_simplified.graphml",
    },
    "Lapland": {
        # Large Lapland area with Rovaniemi as the central hub.
        "center_lat": 66.5039,
        "center_lon": 25.7294,
        "width_km": 90,
        "height_km": 80,
        "graph_file": "lapland_polygon_simplified.graphml",
    },
    "Almelo": {
        # Almelo, including the city and surrounding main roads in Twente.
        "center_lat": 52.3567,
        "center_lon": 6.6625,
        "width_km": 14,
        "height_km": 14,
        "graph_file": "almelo_polygon_simplified.graphml",
    },
    "Houten": {
        "center_lat": 52.0266,
        "center_lon": 5.1788,
        "width_km": 12,
        "height_km": 12,
        "graph_file": "houten_polygon_simplified.graphml",
    },
    "Sneek": {
        # Sneek, including the city, surrounding polders, and the N7 corridor.
        "center_lat": 53.0333,
        "center_lon": 5.6583,
        "width_km": 14,
        "height_km": 14,
        "graph_file": "sneek_polygon_simplified.graphml",
    },
    "Varėna": {
        # Varėna, Lithuania, including the town and surrounding roads.
        "center_lat": 54.21546,
        "center_lon": 24.57538,
        "width_km": 15,
        "height_km": 15,
        "graph_file": "varena_polygon_simplified.graphml",
    },
    "Nicosia": {
        # Nicosia / Lefkoşa, Cyprus, including the city and suburbs.
        "center_lat": 35.1856,
        "center_lon": 33.3823,
        "width_km": 15,
        "height_km": 15,
        "graph_file": "nicosia_polygon_simplified.graphml",
    },
}

if CITY not in REGION_CONFIGS:
    raise ValueError(f"Unknown city or region: {CITY}")

config = REGION_CONFIGS[CITY]
CENTER_LAT = config["center_lat"]
CENTER_LON = config["center_lon"]
WIDTH_KM = config["width_km"]
HEIGHT_KM = config["height_km"]
GRAPH_FILE = config["graph_file"]
# endregion


# region Polygon Construction
# This block converts the configured center point and dimensions into a rectangular polygon.
lat_deg_per_km = 1 / 111
lon_deg_per_km = 1 / (111 * math.cos(math.radians(CENTER_LAT)))

north = CENTER_LAT + (HEIGHT_KM / 2) * lat_deg_per_km
south = CENTER_LAT - (HEIGHT_KM / 2) * lat_deg_per_km
east = CENTER_LON + (WIDTH_KM / 2) * lon_deg_per_km
west = CENTER_LON - (WIDTH_KM / 2) * lon_deg_per_km

polygon = Polygon(
    [
        (west, south),
        (east, south),
        (east, north),
        (west, north),
    ]
)
# endregion


# region Road Filter
# This block defines which OpenStreetMap road categories are included in the network.
custom_filter = (
    '["highway"~'
    '"motorway|trunk|primary|secondary|tertiary|'
    'residential|unclassified|service|living_street|road|track"]'
)
# endregion


# region Network Download
# This block downloads the road network inside the configured polygon.
print(f"Downloading road network for {CITY}...")
G = ox.graph_from_polygon(
    polygon,
    network_type="all",
    custom_filter=custom_filter,
    simplify=False,
)
print(f"Before simplification: nodes={len(G.nodes)}, edges={len(G.edges)}")
# endregion


# region Network Simplification
# This block simplifies the road-network topology while preserving road geometry.
print("Simplifying network topology while preserving road geometry...")
G = ox.simplify_graph(G)
print(f"After simplification: nodes={len(G.nodes)}, edges={len(G.edges)}")
# endregion


# region Speed and Travel Time Attributes
# This block adds road speed and travel-time attributes to the graph edges.
G = ox.add_edge_speeds(G)        # km/h
G = ox.add_edge_travel_times(G)  # seconds
# endregion


# region Save Graph
# This block saves the prepared road network as a GraphML file for the CASEVAC simulation.
ox.save_graphml(G, GRAPH_FILE)
print(f"Network saved as {GRAPH_FILE}")
# endregion
