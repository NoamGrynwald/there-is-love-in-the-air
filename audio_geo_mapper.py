import geopandas as gpd
import pandas as pd
import folium

from shapely.geometry import (
    Point,
    Polygon,
    MultiPolygon
)

from pathlib import Path


# ============================================================
# Audio -> GIS layer mapping
# ============================================================

AUDIO_LAYER_MAPPING = {
    "Train": "railways",
    "Rail transport": "railways",

    "Car": "roads",
    "Vehicle": "roads",

    "Dog": "parks",
    "Animal": "parks",
    "Bird": "parks",

    "Waterfall": "water",
    "Stream": "water",

    "Speech": "buildings",

    "Cattle": "agriculture",
    "Livestock": "agriculture"
}


# ============================================================
# Parse classifier output
# ============================================================

def parse_audio_classes(classification_str):
    """
    Example:

    Train (0.82); Car (0.61)

    ->
    ["Train", "Car"]
    """

    if pd.isna(classification_str):
        return []

    if not classification_str:
        return []

    classes = []

    for item in classification_str.split(";"):

        item = item.strip()

        if "(" in item:
            item = item.split("(")[0].strip()

        if item:
            classes.append(item)

    return classes


# ============================================================
# Get GIS layers from detected sounds
# ============================================================

def get_relevant_layers(classification_str):

    classes = parse_audio_classes(
        classification_str
    )

    layers = set()

    for cls in classes:

        if cls in AUDIO_LAYER_MAPPING:
            layers.add(
                AUDIO_LAYER_MAPPING[cls]
            )

    return list(layers)


# ============================================================
# Create search buffer
# ============================================================

def create_buffer(
        center_lat,
        center_lon,
        radius_meters
):

    point = gpd.GeoSeries(
        [Point(center_lon, center_lat)],
        crs="EPSG:4326"
    )

    point_3857 = point.to_crs(
        "EPSG:3857"
    )

    buffer_3857 = point_3857.buffer(
        radius_meters
    )

    buffer_4326 = (
        gpd.GeoSeries(
            buffer_3857,
            crs="EPSG:3857"
        )
        .to_crs("EPSG:4326")
    )

    return buffer_4326.iloc[0]


# ============================================================
# Load GIS layer
# ============================================================

def load_layer(layer_path):

    if not Path(layer_path).exists():

        print(
            f"[WARNING] Missing layer: {layer_path}"
        )

        return None

    try:
        return gpd.read_file(layer_path)

    except Exception as e:

        print(
            f"[ERROR] Failed loading "
            f"{layer_path}: {e}"
        )

        return None


# ============================================================
# Find geometries inside buffer
# ============================================================

def find_candidate_geometries(
        classification_str,
        center_lat,
        center_lon,
        radius_meters,
        layer_folder="gis_layers"
):

    buffer_geom = create_buffer(
        center_lat,
        center_lon,
        radius_meters
    )

    relevant_layers = get_relevant_layers(
        classification_str
    )

    print("\nDetected classes:")
    print(
        parse_audio_classes(
            classification_str
        )
    )

    print("\nRelevant GIS layers:")
    print(relevant_layers)

    candidates = []

    for layer_name in relevant_layers:

        layer_path = (
            f"{layer_folder}/"
            f"{layer_name}.geojson"
        )

        gdf = load_layer(
            layer_path
        )

        if gdf is None:
            continue

        if gdf.crs is None:
            gdf.set_crs(
                "EPSG:4326",
                inplace=True
            )

        try:

            inside = gdf[
                gdf.intersects(
                    buffer_geom
                )
            ]

            print(
                f"{layer_name}: "
                f"{len(inside)} geometries"
            )

            if len(inside) > 0:
                candidates.append(
                    (
                        layer_name,
                        inside
                    )
                )

        except Exception as e:

            print(
                f"Spatial query failed "
                f"for {layer_name}: {e}"
            )

    return candidates, buffer_geom


# ============================================================
# Draw polygon
# ============================================================

def draw_polygon(
        m,
        polygon,
        color="red",
        fill_opacity=0.35
):

    if isinstance(
            polygon,
            Polygon
    ):

        coords = [
            [lat, lon]
            for lon, lat
            in polygon.exterior.coords
        ]

        folium.Polygon(
            locations=coords,
            color=color,
            weight=4,
            fill=True,
            fill_opacity=fill_opacity
        ).add_to(m)

    elif isinstance(
            polygon,
            MultiPolygon
    ):

        for poly in polygon.geoms:

            coords = [
                [lat, lon]
                for lon, lat
                in poly.exterior.coords
            ]

            folium.Polygon(
                locations=coords,
                color=color,
                weight=4,
                fill=True,
                fill_opacity=fill_opacity
            ).add_to(m)


# ============================================================
# Create map
# ============================================================

def create_map(
        center_lat,
        center_lon,
        candidates,
        buffer_geom,
        output_html="audio_map.html"
):

    m = folium.Map(
        location=[
            center_lat,
            center_lon
        ],
        zoom_start=14
    )

    # --------------------------------------------------
    # Draw search area
    # --------------------------------------------------

    draw_polygon(
        m,
        buffer_geom,
        color="red",
        fill_opacity=0.25
    )

    # --------------------------------------------------
    # Draw GIS layers
    # --------------------------------------------------

    for layer_name, gdf in candidates:

        folium.GeoJson(
            gdf.__geo_interface__,
            name=layer_name,
            style_function=lambda x: {
                "color": "blue",
                "weight": 3,
                "fillOpacity": 0.5
            }
        ).add_to(m)

    # --------------------------------------------------
    # Center marker
    # --------------------------------------------------

    folium.Marker(
        [
            center_lat,
            center_lon
        ],
        popup="Audio Center"
    ).add_to(m)

    # --------------------------------------------------
    # Auto zoom
    # --------------------------------------------------

    minx, miny, maxx, maxy = (
        buffer_geom.bounds
    )

    m.fit_bounds(
        [
            [miny, minx],
            [maxy, maxx]
        ]
    )

    folium.LayerControl().add_to(m)

    m.save(output_html)

    print(
        f"\nMap saved to "
        f"{output_html}"
    )

    print(
        "\nBuffer bounds:"
    )

    print(
        buffer_geom.bounds
    )


# ============================================================
# Main API
# ============================================================

def generate_audio_location_map(
        classification_str,
        center_lat,
        center_lon,
        radius_meters=1000,
        output_html="audio_map.html"
):

    candidates, buffer_geom = (
        find_candidate_geometries(
            classification_str,
            center_lat,
            center_lon,
            radius_meters
        )
    )

    create_map(
        center_lat,
        center_lon,
        candidates,
        buffer_geom,
        output_html
    )


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    test_classification = (
        "Train (0.91); "
        "Bird (0.72); "
        "Stream (0.65)"
    )

    generate_audio_location_map(
        classification_str=test_classification,
        center_lat=32.0853,
        center_lon=34.7818,
        radius_meters=1000,
        output_html="audio_map.html"
    )