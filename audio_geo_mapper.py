"""
audio_geo_mapper.py
====================
Maps audio classification labels to real geographic polygon geometries
using OpenStreetMap (Nominatim) and Shapely.

For each audio sample, generates an interactive Folium map showing:
  - A starter buffer circle around the recording location
  - Shapely polygons for each detected sound class (OSM-based where possible,
    procedurally generated otherwise)
"""

import re
import time
import math
import warnings
from typing import Optional

import requests
import numpy as np
import folium
import folium.plugins
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import geopandas as gpd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# NOMINATIM QUERY MAP
# Maps AudioSet label groups → OSM Nominatim search queries + amenity/tag hints
# ─────────────────────────────────────────────────────────────────────────────

LABEL_TO_OSM_QUERY = {
    # Human speech / voices → public spaces
    "Speech": ("public square", "place"),
    "Male speech, man speaking": ("community centre", "amenity"),
    "Female speech, woman speaking": ("community centre", "amenity"),
    "Child speech, kid speaking": ("playground", "leisure"),
    "Conversation": ("cafe", "amenity"),
    "Babbling": ("nursery", "amenity"),
    "Shout": ("stadium", "leisure"),
    "Screaming": ("hospital emergency", "amenity"),
    "Whispering": ("library", "amenity"),
    "Laughter": ("park", "leisure"),
    "Baby laughter": ("playground", "leisure"),
    "Giggle": ("park", "leisure"),
    "Crying, sobbing": ("hospital", "amenity"),
    "Baby cry, infant cry": ("hospital", "amenity"),
    "Singing": ("concert hall", "amenity"),
    "Choir": ("church", "amenity"),
    "Chant": ("mosque", "amenity"),
    "Mantra": ("place of worship", "amenity"),
    "Rapping": ("music venue", "amenity"),
    "Humming": ("music school", "amenity"),

    # Running / walking / movement
    "Run": ("athletics track", "leisure"),
    "Shuffle": ("corridor", None),
    "Walk, footsteps": ("pedestrian street", "highway"),

    # Eating / body sounds
    "Chewing, mastication": ("restaurant", "amenity"),
    "Frying (food)": ("restaurant kitchen", "amenity"),

    # Crowd / audience
    "Cheering": ("stadium", "leisure"),
    "Applause": ("theatre", "amenity"),
    "Chatter": ("market", "amenity"),
    "Crowd": ("town square", "place"),
    "Children playing": ("playground", "leisure"),

    # Animals – domestic
    "Dog": ("dog park", "leisure"),
    "Bark": ("dog park", "leisure"),
    "Howl": ("forest", "landuse"),
    "Cat": ("residential area", "landuse"),
    "Purr": ("residential area", "landuse"),
    "Meow": ("residential area", "landuse"),

    # Animals – farm / livestock
    "Livestock, farm animals, working animals": ("farm", "landuse"),
    "Horse": ("equestrian", "leisure"),
    "Cattle, bovinae": ("farm", "landuse"),
    "Pig": ("farm", "landuse"),
    "Goat": ("farm", "landuse"),
    "Sheep": ("farm", "landuse"),
    "Fowl": ("farm", "landuse"),
    "Chicken, rooster": ("farm", "landuse"),
    "Turkey": ("farm", "landuse"),
    "Duck": ("pond", "natural"),
    "Goose": ("pond", "natural"),

    # Animals – wild
    "Roaring cats (lions, tigers)": ("zoo", "tourism"),
    "Bird": ("nature reserve", "leisure"),
    "Bird vocalization, bird call, bird song": ("nature reserve", "leisure"),
    "Owl": ("forest", "landuse"),
    "Insect": ("meadow", "landuse"),
    "Cricket": ("meadow", "landuse"),
    "Mosquito": ("wetland", "natural"),
    "Bee, wasp, etc.": ("garden", "leisure"),
    "Frog": ("wetland", "natural"),
    "Snake": ("grassland", "natural"),
    "Whale vocalization": ("ocean", "natural"),

    # Music – venues
    "Music": ("concert hall", "amenity"),
    "Musical instrument": ("music school", "amenity"),
    "Guitar": ("music venue", "amenity"),
    "Piano": ("concert hall", "amenity"),
    "Drum kit": ("music studio", "amenity"),
    "Orchestra": ("philharmonic hall", "amenity"),
    "Organ": ("church", "amenity"),
    "Choir": ("church", "amenity"),
    "Bagpipes": ("park", "leisure"),
    "Shofar": ("synagogue", "amenity"),

    # Music genres
    "Pop music": ("music venue", "amenity"),
    "Hip hop music": ("music venue", "amenity"),
    "Rock music": ("music venue", "amenity"),
    "Heavy metal": ("music venue", "amenity"),
    "Jazz": ("jazz club", "amenity"),
    "Classical music": ("concert hall", "amenity"),
    "Opera": ("opera house", "amenity"),
    "Electronic music": ("nightclub", "amenity"),
    "House music": ("nightclub", "amenity"),
    "Techno": ("nightclub", "amenity"),
    "Dubstep": ("nightclub", "amenity"),
    "Reggae": ("music venue", "amenity"),
    "Country": ("music venue", "amenity"),
    "Gospel music": ("church", "amenity"),
    "Christian music": ("church", "amenity"),
    "Middle Eastern music": ("cultural centre", "amenity"),
    "Lullaby": ("residential area", "landuse"),
    "Dance music": ("nightclub", "amenity"),
    "Wedding music": ("wedding venue", "amenity"),
    "Christmas music": ("church", "amenity"),

    # Nature / weather
    "Wind": ("open field", "landuse"),
    "Rustling leaves": ("forest", "landuse"),
    "Thunderstorm": ("open area", "landuse"),
    "Thunder": ("open area", "landuse"),
    "Water": ("river", "waterway"),
    "Rain": ("park", "leisure"),
    "Stream": ("stream", "waterway"),
    "Waterfall": ("waterfall", "natural"),
    "Ocean": ("beach", "natural"),
    "Waves, surf": ("beach", "natural"),
    "Fire": ("fire station", "amenity"),

    # Vehicles – road
    "Vehicle": ("road", "highway"),
    "Car": ("parking lot", "amenity"),
    "Car alarm": ("parking lot", "amenity"),
    "Truck": ("highway", "highway"),
    "Bus": ("bus station", "amenity"),
    "Motorcycle": ("road", "highway"),
    "Traffic noise, roadway noise": ("intersection", "highway"),
    "Emergency vehicle": ("hospital", "amenity"),
    "Police car (siren)": ("police station", "amenity"),
    "Ambulance (siren)": ("hospital", "amenity"),
    "Fire engine, fire truck (siren)": ("fire station", "amenity"),
    "Ice cream truck, ice cream van": ("park", "leisure"),

    # Vehicles – rail
    "Train": ("railway station", "railway"),
    "Train whistle": ("railway station", "railway"),
    "Subway, metro, underground": ("subway station", "railway"),
    "Rail transport": ("railway", "railway"),

    # Vehicles – air
    "Aircraft": ("airport", "aeroway"),
    "Helicopter": ("helipad", "aeroway"),
    "Jet engine": ("airport", "aeroway"),

    # Vehicles – water
    "Boat, Water vehicle": ("marina", "leisure"),
    "Motorboat, speedboat": ("marina", "leisure"),
    "Ship": ("port", "industrial"),

    # Construction / tools
    "Chainsaw": ("forest", "landuse"),
    "Lawn mower": ("garden", "leisure"),
    "Jackhammer": ("construction site", "landuse"),
    "Drill": ("construction site", "landuse"),
    "Hammer": ("construction site", "landuse"),
    "Power tool": ("construction site", "landuse"),
    "Sawing": ("sawmill", "industrial"),

    # Gunfire / explosions
    "Explosion": ("quarry", "landuse"),
    "Gunshot, gunfire": ("shooting range", "leisure"),
    "Machine gun": ("military", "landuse"),
    "Artillery fire": ("military", "landuse"),
    "Fireworks": ("park", "leisure"),

    # Indoor / domestic
    "Door": ("residential building", "building"),
    "Doorbell": ("residential building", "building"),
    "Typing": ("office", "office"),
    "Computer keyboard": ("office", "office"),
    "Telephone": ("office", "office"),
    "Alarm": ("office building", "building"),
    "Vacuum cleaner": ("residential area", "landuse"),
    "Microwave oven": ("kitchen", "amenity"),
    "Blender": ("kitchen", "amenity"),
    "Toilet flush": ("public toilet", "amenity"),
    "Hair dryer": ("hair salon", "shop"),
    "Printer": ("office", "office"),
    "Camera": ("photography studio", "amenity"),
    "Cash register": ("supermarket", "shop"),
    "Sewing machine": ("textile factory", "industrial"),
    "Air conditioning": ("office building", "building"),
    "Clock": ("town hall", "amenity"),

    # Environment tags
    "Inside, small room": ("room", None),
    "Inside, large room or hall": ("concert hall", "amenity"),
    "Inside, public space": ("shopping mall", "shop"),
    "Outside, urban or manmade": ("city square", "place"),
    "Outside, rural or natural": ("national park", "leisure"),

    # Sports
    "Basketball bounce": ("basketball court", "leisure"),

    # TV / Radio
    "Television": ("broadcast studio", "amenity"),
    "Radio": ("radio station", "office"),
}

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR MAP  (label category → hex colour)
# ─────────────────────────────────────────────────────────────────────────────

def _category_color(label: str) -> str:
    label_lower = label.lower()
    if any(w in label_lower for w in ["speech", "voice", "singing", "laugh", "cry", "shout", "scream"]):
        return "#4A90D9"   # blue – human voice
    if any(w in label_lower for w in ["dog", "cat", "horse", "bird", "animal", "pig", "cow", "sheep", "frog", "insect", "bee", "snake", "whale"]):
        return "#27AE60"   # green – animals
    if any(w in label_lower for w in ["music", "guitar", "piano", "drum", "orchestra", "jazz", "rock", "pop", "hip hop", "classical", "opera", "electronic", "lullaby", "song", "choir"]):
        return "#8E44AD"   # purple – music
    if any(w in label_lower for w in ["car", "truck", "train", "aircraft", "boat", "vehicle", "motorcycle", "bus", "helicopter", "ship", "bicycle", "skateboard"]):
        return "#E67E22"   # orange – transport
    if any(w in label_lower for w in ["rain", "thunder", "wind", "water", "ocean", "stream", "fire", "waterfall", "wave"]):
        return "#1ABC9C"   # teal – nature / weather
    if any(w in label_lower for w in ["gun", "explosion", "artillery", "firework", "bomb"]):
        return "#E74C3C"   # red – explosions / weapons
    if any(w in label_lower for w in ["drill", "jackhammer", "chainsaw", "hammer", "saw", "mower", "power tool", "construction"]):
        return "#F39C12"   # amber – construction
    if any(w in label_lower for w in ["door", "typing", "alarm", "telephone", "vacuum", "clock", "printer", "camera", "keyboard"]):
        return "#95A5A6"   # grey – indoor / domestic
    return "#BDC3C7"       # light grey – other


# ─────────────────────────────────────────────────────────────────────────────
# GEO HELPERS
# ─────────────────────────────────────────────────────────────────────────────

EARTH_RADIUS_M = 6_371_000


def _offset_latlon(lat: float, lon: float, distance_m: float, bearing_deg: float):
    """Move (lat, lon) by distance_m metres in direction bearing_deg."""
    bearing = math.radians(bearing_deg)
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    d = distance_m / EARTH_RADIUS_M

    lat2 = math.asin(
        math.sin(lat_r) * math.cos(d) +
        math.cos(lat_r) * math.sin(d) * math.cos(bearing)
    )
    lon2 = lon_r + math.atan2(
        math.sin(bearing) * math.sin(d) * math.cos(lat_r),
        math.cos(d) - math.sin(lat_r) * math.sin(lat2)
    )
    return math.degrees(lat2), math.degrees(lon2)


def _circle_polygon(lat: float, lon: float, radius_m: float, n_pts: int = 64) -> Polygon:
    """Create a Shapely Polygon approximating a circle on the Earth's surface."""
    coords = []
    for i in range(n_pts):
        bearing = 360 * i / n_pts
        rlat, rlon = _offset_latlon(lat, lon, radius_m, bearing)
        coords.append((rlon, rlat))   # Shapely uses (x=lon, y=lat)
    coords.append(coords[0])
    return Polygon(coords)


def _procedural_polygon(
    center_lat: float,
    center_lon: float,
    label: str,
    radius_meters: float,
    seed: int = 0,
) -> Polygon:
    """
    Generate a plausible irregular Shapely polygon when no OSM match is found.
    The shape type depends on label category.
    """
    rng = np.random.default_rng(seed + hash(label) % (2**31))
    label_lower = label.lower()

    # Choose a sub-radius based on expected real-world footprint
    if any(w in label_lower for w in ["stadium", "airport", "forest", "park", "farm", "ocean", "beach"]):
        r = radius_meters * rng.uniform(0.25, 0.45)
    elif any(w in label_lower for w in ["building", "house", "room", "church", "school"]):
        r = radius_meters * rng.uniform(0.03, 0.08)
    elif any(w in label_lower for w in ["road", "street", "highway", "rail", "train"]):
        r = radius_meters * rng.uniform(0.15, 0.30)
    else:
        r = radius_meters * rng.uniform(0.08, 0.20)

    # Offset the centre slightly
    offset_dist = radius_meters * rng.uniform(0.05, 0.35)
    offset_bear = rng.uniform(0, 360)
    c_lat, c_lon = _offset_latlon(center_lat, center_lon, offset_dist, offset_bear)

    # Build irregular polygon
    n_verts = int(rng.integers(6, 14))
    angles = np.sort(rng.uniform(0, 2 * math.pi, n_verts))
    radii  = r * rng.uniform(0.6, 1.0, n_verts)

    coords = []
    for angle, radius in zip(angles, radii):
        bearing = math.degrees(angle)
        rlat, rlon = _offset_latlon(c_lat, c_lon, radius, bearing)
        coords.append((rlon, rlat))
    coords.append(coords[0])

    poly = Polygon(coords)
    return poly if poly.is_valid else poly.buffer(0)


# ─────────────────────────────────────────────────────────────────────────────
# NOMINATIM LOOKUP
# ─────────────────────────────────────────────────────────────────────────────

_nominatim_cache: dict = {}

def _nominatim_search(
    query: str,
    near_lat: float,
    near_lon: float,
    radius_meters: float,
) -> Optional[dict]:
    """
    Query Nominatim for a place near (near_lat, near_lon).
    Returns a dict with 'lat', 'lon', 'bbox' or None.
    """
    cache_key = f"{query}|{near_lat:.3f}|{near_lon:.3f}"
    if cache_key in _nominatim_cache:
        return _nominatim_cache[cache_key]

    # Build bounded search box (~radius_meters around centre)
    lat_deg = radius_meters / 111_000
    lon_deg = radius_meters / (111_000 * math.cos(math.radians(near_lat)))
    viewbox = (
        f"{near_lon - lon_deg},{near_lat - lat_deg},"
        f"{near_lon + lon_deg},{near_lat + lat_deg}"
    )

    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "json",
        "limit": 1,
        "viewbox": viewbox,
        "bounded": 1,
        "addressdetails": 0,
    }
    headers = {"User-Agent": "audio-geo-mapper/1.0 (research project)"}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=8)
        time.sleep(1.1)   # Nominatim rate-limit: 1 req/sec
        data = resp.json()
        if data:
            result = data[0]
            out = {
                "lat": float(result["lat"]),
                "lon": float(result["lon"]),
                "bbox": result.get("boundingbox"),   # [min_lat, max_lat, min_lon, max_lon]
                "display_name": result.get("display_name", query),
            }
            _nominatim_cache[cache_key] = out
            return out
    except Exception as e:
        print(f"    [Nominatim] Error for '{query}': {e}")

    _nominatim_cache[cache_key] = None
    return None


def _osm_polygon(
    label: str,
    center_lat: float,
    center_lon: float,
    radius_meters: float,
    seed: int = 0,
) -> tuple[Polygon, str, bool]:
    """
    Return (shapely_polygon, source_description, is_osm).
    Tries Nominatim first; falls back to procedural geometry.
    """
    osm_info = LABEL_TO_OSM_QUERY.get(label)

    if osm_info:
        query, _ = osm_info
        result = _nominatim_search(query, center_lat, center_lon, radius_meters)
        if result:
            bbox = result.get("bbox")
            if bbox:
                min_lat, max_lat, min_lon, max_lon = (float(x) for x in bbox)
                poly = Polygon([
                    (min_lon, min_lat),
                    (max_lon, min_lat),
                    (max_lon, max_lat),
                    (min_lon, max_lat),
                    (min_lon, min_lat),
                ])
                desc = result["display_name"]
                return poly, desc, True
            else:
                # Point result → small circle
                r = radius_meters * 0.05
                poly = _circle_polygon(result["lat"], result["lon"], r)
                return poly, result["display_name"], True

    # Fallback: procedural
    poly = _procedural_polygon(center_lat, center_lon, label, radius_meters, seed)
    return poly, f"Procedural geometry for '{label}'", False


# ─────────────────────────────────────────────────────────────────────────────
# PARSE CLASSIFICATION STRING
# ─────────────────────────────────────────────────────────────────────────────

def parse_classification(classification_str: str) -> list[tuple[str, float]]:
    """
    Parse 'Label A (0.85); Label B (0.72)' into [(label, score), ...].
    Sorted descending by score.
    """
    items = []
    for part in classification_str.split(";"):
        part = part.strip()
        m = re.match(r"^(.+?)\s*\((\d+\.\d+)\)$", part)
        if m:
            label = m.group(1).strip()
            score = float(m.group(2))
            items.append((label, score))
    return sorted(items, key=lambda x: x[1], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# POPUP BUILDER  (avoids backslash-in-f-string, compatible with Python < 3.12)
# ─────────────────────────────────────────────────────────────────────────────

def _build_polygon_popup(label: str, score: float, is_osm: bool, source_desc: str) -> str:
    source_html = (
        '<span style="color:green">OpenStreetMap</span>'
        if is_osm else
        '<span style="color:orange">Procedural</span>'
    )
    desc_snippet = source_desc[:120]
    return (
        f"<b>{label}</b><br>"
        f"Confidence: <b>{score:.0%}</b><br>"
        f"Source: {source_html}<br>"
        f"<small>{desc_snippet}</small>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE MAP GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_audio_location_map(
    classification_str: str,
    center_lat: float,
    center_lon: float,
    radius_meters: float = 10_000,
    output_path: str = "audio_map.html",
    sample_id: Optional[str] = None,
) -> str:
    """
    Generate a Folium map for a single audio classification result.

    Parameters
    ----------
    classification_str : str
        Semicolon-separated labels+scores, e.g. "Dog (0.85); Car (0.72)"
    center_lat, center_lon : float
        GPS coordinates of the recording location.
    radius_meters : float
        Search radius and starter buffer size (metres).
    output_path : str
        Where to save the HTML map.
    sample_id : str, optional
        Label shown in map title.

    Returns
    -------
    str
        Path to the saved HTML file.
    """
    parsed = parse_classification(classification_str)
    if not parsed:
        print(f"[WARN] No valid classifications found in: {classification_str!r}")
        return output_path

    title = f"Audio Map – {sample_id}" if sample_id else "Audio Classification Map"
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"  Centre: ({center_lat:.5f}, {center_lon:.5f})  |  Buffer: {radius_meters/1000:.1f} km")
    print(f"  Labels detected: {len(parsed)}")
    print(f"{'='*60}")

    # ── Folium map ──────────────────────────────────────────────────────────
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="CartoDB positron",
    )
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Dark Matter").add_to(m)

    # Starter buffer circle (blue, dashed)
    folium.Circle(
        location=[center_lat, center_lon],
        radius=radius_meters,
        color="#2980B9",
        weight=2,
        fill=True,
        fill_color="#2980B9",
        fill_opacity=0.05,
        dash_array="10,6",
        tooltip=f"Detection buffer: {radius_meters/1000:.1f} km",
        popup=folium.Popup(
            f"<b>Recording location</b><br>"
            f"({center_lat:.5f}, {center_lon:.5f})<br>"
            f"Buffer radius: {radius_meters/1000:.1f} km",
            max_width=250,
        ),
    ).add_to(m)

    # Centre marker
    folium.Marker(
        location=[center_lat, center_lon],
        popup=folium.Popup(f"<b>Recording point</b><br>{center_lat:.5f}, {center_lon:.5f}", max_width=200),
        icon=folium.Icon(color="blue", icon="microphone", prefix="fa"),
        tooltip="Recording location",
    ).add_to(m)

    # ── Feature groups per label ────────────────────────────────────────────
    for idx, (label, score) in enumerate(parsed):
        color = _category_color(label)
        opacity = max(0.25, min(0.65, score * 0.7))

        print(f"  [{idx+1:02d}] {label} ({score:.2f}) → querying OSM …")
        poly, source_desc, is_osm = _osm_polygon(
            label, center_lat, center_lon, radius_meters, seed=idx
        )

        # Convert Shapely (lon, lat) → Folium [lat, lon] list
        if isinstance(poly, MultiPolygon):
            polys = list(poly.geoms)
        else:
            polys = [poly]

        osm_tag = "[OSM]" if is_osm else "[proc]"
        fg_label = folium.FeatureGroup(
            name=f"{osm_tag} {label} ({score:.2f})",
            show=True,
        )

        for shp in polys:
            if shp.is_empty:
                continue
            coords_folium = [[lat, lon] for lon, lat in shp.exterior.coords]
            folium.Polygon(
                locations=coords_folium,
                color=color,
                weight=2,
                fill=True,
                fill_color=color,
                fill_opacity=opacity,
                tooltip=f"{label} | score: {score:.2f} | {'OSM' if is_osm else 'procedural'}",
                popup=folium.Popup(
                    _build_polygon_popup(label, score, is_osm, source_desc),
                    max_width=300,
                ),
            ).add_to(fg_label)

        fg_label.add_to(m)

    # ── Legend ──────────────────────────────────────────────────────────────
    legend_rows = "".join(
        f"<tr>"
        f"<td style='width:14px;height:14px;background:{_category_color(lbl)};border-radius:3px;border:1px solid #aaa'></td>"
        f"<td style='padding-left:6px;font-size:12px'>{lbl} <span style='color:#888'>({sc:.2f})</span></td>"
        f"</tr>"
        for lbl, sc in parsed
    )
    legend_html = f"""
    <div style="
        position: fixed; bottom: 40px; right: 12px; z-index: 1000;
        background: rgba(255,255,255,0.95); border-radius: 8px;
        border: 1px solid #ccc; padding: 12px 14px;
        font-family: sans-serif; max-height: 320px; overflow-y: auto;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.15);
    ">
        <b style='font-size:13px'>🔊 Audio Classifications</b>
        <table style='margin-top:6px;border-collapse:collapse'>
            {legend_rows}
        </table>
        <div style='margin-top:8px;font-size:11px;color:#555'>
            [OSM] = OpenStreetMap geometry &nbsp;|&nbsp; [proc] = Procedural
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # ── Title ───────────────────────────────────────────────────────────────
    title_html = f"""
    <div style="
        position: fixed; top: 12px; left: 50%; transform: translateX(-50%);
        z-index: 1000; background: rgba(255,255,255,0.92);
        border-radius: 6px; border: 1px solid #ccc;
        padding: 8px 18px; font-family: sans-serif;
        font-size: 15px; font-weight: bold;
        box-shadow: 1px 1px 6px rgba(0,0,0,0.12);
    ">{title}</div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    folium.LayerControl(collapsed=False).add_to(m)
    folium.plugins.Fullscreen().add_to(m)
    folium.plugins.MiniMap().add_to(m)

    m.save(output_path)
    print(f"\n  ✅ Map saved → {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# BATCH: one map per DataFrame row
# ─────────────────────────────────────────────────────────────────────────────

def generate_maps_for_dataframe(
    df,
    classification_col: str = "audio_object_classification",
    lat_col: str = "center_lat",
    lon_col: str = "center_lon",
    sample_id_col: str = "sample_id",
    center_lat: Optional[float] = None,
    center_lon: Optional[float] = None,
    radius_meters: float = 10_000,
    output_dir: str = ".",
) -> list[str]:
    """
    Generate one map per row in a DataFrame.

    If the DataFrame has lat/lon columns use those; otherwise fall back to
    the provided center_lat/center_lon defaults.

    Parameters
    ----------
    df : pd.DataFrame
    classification_col : str
        Column containing classification strings.
    lat_col, lon_col : str
        Column names for per-row coordinates (optional, falls back to defaults).
    sample_id_col : str
        Column used for map title and filename.
    center_lat, center_lon : float
        Fallback coordinates when per-row columns are absent.
    radius_meters : float
        Buffer radius.
    output_dir : str
        Directory where HTML files are written.

    Returns
    -------
    list[str]
        Paths to all generated HTML maps.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    paths = []
    for i, row in df.iterrows():
        sample_id = str(row.get(sample_id_col, f"sample_{i}"))
        classification_str = row.get(classification_col, "")

        if not classification_str or str(classification_str).startswith("ERROR"):
            print(f"[SKIP] Row {i} ({sample_id}): invalid classification")
            continue

        # Resolve coordinates
        row_lat = row.get(lat_col, center_lat)
        row_lon = row.get(lon_col, center_lon)
        if row_lat is None or row_lon is None:
            print(f"[SKIP] Row {i} ({sample_id}): no coordinates")
            continue

        safe_id = re.sub(r"[^\w\-]", "_", sample_id)
        out_path = os.path.join(output_dir, f"audio_map_{safe_id}.html")

        generate_audio_location_map(
            classification_str=str(classification_str),
            center_lat=float(row_lat),
            center_lon=float(row_lon),
            radius_meters=radius_meters,
            output_path=out_path,
            sample_id=sample_id,
        )
        paths.append(out_path)

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Quick standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_classification = (
        "Dog (0.88); Car (0.74); Bird vocalization, bird call, bird song (0.61); "
        "Traffic noise, roadway noise (0.52); Rain (0.43); Crowd (0.38)"
    )
    generate_audio_location_map(
        classification_str=test_classification,
        center_lat=32.07242,
        center_lon=34.78305,
        radius_meters=10_000,
        output_path="audio_map_test.html",
        sample_id="test_sample_001",
    )