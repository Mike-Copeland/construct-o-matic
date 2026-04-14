# process_plans_v7.py (Digital Twin - Multi-Geometry Extraction)

import google.generativeai as genai
import json
import os
import math
import re
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pyproj import CRS, Transformer
from typing import Optional, Tuple, List, Dict, Any
import simplekml

# --- Configuration ---
MODEL_NAME = "gemini-3.1-pro-preview"

# --- Category Definitions (color-coded layers) ---
CATEGORIES = {
    "centerline":       {"color": "#FF0000", "kml_color": "ff0000ff", "width": 3},
    "drainage":         {"color": "#0000FF", "kml_color": "ffff0000", "width": 3},
    "signing":          {"color": "#FFD700", "kml_color": "ff00d7ff", "width": 2},
    "pavement_marking": {"color": "#FFFFFF", "kml_color": "ffffffff", "width": 2},
    "curb_gutter":      {"color": "#808080", "kml_color": "ff808080", "width": 4},
    "sidewalk":         {"color": "#C0C0C0", "kml_color": "ffc0c0c0", "width": 4},
    "guardrail":        {"color": "#FF4500", "kml_color": "ff0045ff", "width": 4},
    "approach":         {"color": "#8B4513", "kml_color": "ff13458b", "width": 2},
    "erosion_control":  {"color": "#228B22", "kml_color": "ff228b22", "width": 3},
    "traffic_control":  {"color": "#FF8C00", "kml_color": "ff008cff", "width": 2},
    "survey_monument":  {"color": "#FF00FF", "kml_color": "ffff00ff", "width": 2},
    "utility":          {"color": "#800080", "kml_color": "ff800080", "width": 2},
    "roadway_surface":  {"color": "#333333", "kml_color": "ff333333", "width": 2},
    "right_of_way":     {"color": "#00CED1", "kml_color": "ffd1ce00", "width": 2},
    "other":            {"color": "#696969", "kml_color": "ff696969", "width": 2},
}

# --- Extraction Configurations ---
# Each config has 'match_keywords' for fuzzy matching against sheet titles from the inventory.
# A config matches if ANY keyword appears (case-insensitive) in ANY sheet title.
FEATURE_SCHEMA_BLOCK = (
    'Return a JSON array where each object has EXACTLY these keys:\n'
    '{"name":"<item code>","description":"<description>","category":"<category>",'
    '"geometry_type":"<point or line>","start_station":"<e.g. 116+20>",'
    '"end_station":"<end station or null>","offset":"<e.g. 35\' Lt.>",'
    '"side":"<Lt or Rt or CL>","notes":"<additional info>"}\n'
)

EXTRACTION_CONFIGS = [
    {
        "name": "Roadway Summary",
        "match_keywords": ["roadway summary"],
        "prompt": (
            "From the ROADWAY SUMMARY sheets, extract ALL pay items with station ranges. "
            "These tables list every construction item, its quantity, and where it applies.\n\n"
            "Include: excavation, embankment, aggregate base, HMA paving, seeding, "
            "removal items, fencing, guardrail, concrete items, and all other listed items.\n\n"
            "For items with a station range (e.g. Sta 116+21 to 256+15), set geometry_type to 'line'.\n"
            "For items at a single location, set geometry_type to 'point'.\n"
            "Set category to the best fit: curb_gutter, sidewalk, guardrail, drainage, "
            "pavement_marking, erosion_control, signing, approach, roadway_surface, or other.\n\n"
            + FEATURE_SCHEMA_BLOCK +
            "Extract EVERY line item. Do not skip any."
        ),
    },
    {
        "name": "Pipe & Culvert Summary",
        "match_keywords": ["pipe", "culvert"],
        "prompt": (
            "From the PIPE CULVERT SUMMARY or PIPE SUMMARY sheets, extract ALL pipes and culverts.\n\n"
            "Each pipe/culvert has a station, size, length, material, and type. "
            "These are POINT features at their station location.\n\n"
            "Set category to 'drainage' for all items.\n\n"
            + FEATURE_SCHEMA_BLOCK +
            "Include pipe size, material, and length in the notes field."
        ),
    },
    {
        "name": "Plan & Profile Features",
        "match_keywords": ["plan & profile", "plan sheet", "plan/profile", "plan/ profile"],
        "prompt": (
            "From the PLAN & PROFILE sheets, extract ALL roadside features "
            "referenced to mainline alignment stationing. Include curb & gutter, sidewalk, guardrail, "
            "barrier, approaches (driveways), retaining walls, and any other constructed features.\n\n"
            "CRITICAL: For LINEAR features (curb & gutter, sidewalk, guardrail, barrier), "
            "provide BOTH start_station AND end_station.\n"
            "Approaches/driveways are POINT features.\n"
            "Set category to: curb_gutter, sidewalk, guardrail, approach, or other.\n\n"
            + FEATURE_SCHEMA_BLOCK +
            "Extract EVERY feature shown. Do not skip any."
        ),
    },
    {
        "name": "Drainage Features",
        "match_keywords": ["drainage"],
        "prompt": (
            "From the DRAINAGE plan sheets, extract ALL drainage structures and features.\n\n"
            "Include: inlets, manholes, catch basins, pipe runs, ponds, swales, ditches, culverts, headwalls.\n"
            "Structures (inlets, manholes) = POINT. Pipe runs/swales/ditches = LINE with start/end station.\n"
            "Set category to 'drainage'.\n\n"
            + FEATURE_SCHEMA_BLOCK +
            "Include pipe size, material, and connected structures in notes."
        ),
    },
    {
        "name": "Signing Features",
        "match_keywords": ["sign"],
        "prompt": (
            "From the SIGNING sheets, Sign Summary table, and Sign & Pavement Marking Plan sheets, "
            "extract ALL signs referenced to mainline alignment stationing.\n\n"
            "Each sign is a POINT feature. Include MUTCD code if shown.\n"
            "Set category to 'signing'.\n\n"
            + FEATURE_SCHEMA_BLOCK +
            "Include facing direction, mounting type, new/existing/remove in notes."
        ),
    },
    {
        "name": "Pavement Marking Features",
        "match_keywords": ["pavement mark", "striping", "marking plan"],
        "prompt": (
            "From the PAVEMENT MARKING / STRIPING sheets, extract ALL pavement markings.\n\n"
            "Striping lines (center line, edge line, lane line) = LINE with start/end station.\n"
            "Symbols (arrows, crosswalks, stop bars) = POINT.\n"
            "Set category to 'pavement_marking'.\n\n"
            + FEATURE_SCHEMA_BLOCK +
            "Include color, width, pattern (solid/skip/broken) in notes."
        ),
    },
    {
        "name": "Erosion Control Features",
        "match_keywords": ["erosion", "swppp"],
        "prompt": (
            "From the EROSION CONTROL / SWPPP sheets, extract ALL erosion control features.\n\n"
            "Silt fence, wattles, erosion blanket = LINE with start/end station.\n"
            "Inlet protection, check dams, sediment traps, construction entrances = POINT.\n"
            "Set category to 'erosion_control'.\n\n"
            + FEATURE_SCHEMA_BLOCK
        ),
    },
    {
        "name": "Traffic Control Features",
        "match_keywords": ["traffic control", "traffic plan"],
        "prompt": (
            "From the TRAFFIC CONTROL / TRAFFIC PLAN sheets, extract ALL temporary traffic control features.\n\n"
            "Include: temporary signs, barricades, flagging stations, temporary striping, arrow boards.\n"
            "Most are POINT features unless they define a zone (LINE).\n"
            "Set category to 'traffic_control'.\n\n"
            + FEATURE_SCHEMA_BLOCK
        ),
    },
    {
        "name": "Utility Features",
        "match_keywords": ["utility"],
        "prompt": (
            "From the UTILITY PLANS sheets, extract ALL utility features referenced to mainline stationing.\n\n"
            "Include: utility relocations, crossings, poles, vaults, manholes, conduit runs.\n"
            "Individual structures = POINT. Conduit/pipe runs = LINE with start/end station.\n"
            "Set category to 'utility'.\n\n"
            + FEATURE_SCHEMA_BLOCK +
            "Include utility owner/type (electric, gas, water, telecom) in notes."
        ),
    },
]

# --- Helper Classes ---

class WorkflowError(Exception):
    """Custom exception for workflow-specific errors."""
    pass

# --- API Utilities ---

def get_api_key():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        try:
            import getpass
            api_key = getpass.getpass("Please enter your Google AI API Key: ")
        except ImportError:
            api_key = input("Please enter your Google AI API Key: ")
    if not api_key:
        print("API Key is required. Exiting.")
        sys.exit(1)
    return api_key

_uploaded_file_cache = {}

def call_gemini_api(pdf_path: str, prompt: str, is_json: bool = True) -> Any:
    print(f"  > Calling Gemini API ('{MODEL_NAME}') for: '{prompt[:60]}...'")
    try:
        # Reuse uploaded file if same path
        if pdf_path in _uploaded_file_cache:
            uploaded_file = _uploaded_file_cache[pdf_path]
        else:
            uploaded_file = genai.upload_file(path=pdf_path, display_name="Plan Set PDF")
            _uploaded_file_cache[pdf_path] = uploaded_file
            print(f"  > Successfully uploaded file: {uploaded_file.display_name}")

        model = genai.GenerativeModel(MODEL_NAME)
        generation_config = {"temperature": 1.0}
        if is_json:
            generation_config["response_mime_type"] = "application/json"
        response = model.generate_content(
            [prompt, uploaded_file],
            generation_config=generation_config
        )
        cleaned_text = re.sub(r'^```json\s*|\s*```$', '', response.text.strip(), flags=re.MULTILINE)
        return json.loads(cleaned_text)
    except Exception as e:
        print(f"  > An error occurred during the API call: {e}")
        raise WorkflowError(f"Failed to get a valid response from the Gemini API: {e}")

# --- Parsing Functions (reused from v6) ---

def parse_station_string(station_val) -> Optional[float]:
    if station_val is None:
        return None
    if isinstance(station_val, (int, float)):
        return float(station_val)
    if isinstance(station_val, str):
        try:
            return float(station_val.replace('+', '')) if '+' in station_val else float(station_val)
        except (ValueError, TypeError):
            return None
    return None

def parse_station_value_from_props(prop: dict) -> Optional[float]:
    station_prop = prop.get('station') or prop.get('sta') or prop.get('STATION')
    if station_prop:
        parsed_station = parse_station_string(station_prop)
        if parsed_station:
            return parsed_station
    name = str(prop.get('name', ''))
    description = str(prop.get('description', ''))
    station_match = re.search(r'STA\.?\s*(\d{1,4})\+(\d{2}(\.\d+)?)', name + description, re.IGNORECASE)
    if station_match:
        return float(f"{station_match.group(1)}{station_match.group(2)}")
    try:
        return float(name)
    except (ValueError, TypeError):
        return None

def parse_offset_value(item: dict) -> Tuple[Optional[float], bool]:
    offset_str = str(item.get('offset', '')).lower().strip()
    dist_cl = item.get('distance_centerline')
    direction = -1 if 'lt' in offset_str or 'left' in offset_str else 1
    distance_val = None
    try:
        distance_val = float(dist_cl)
    except (ValueError, TypeError):
        if offset_str:
            match = re.search(r'(\d+(\.\d+)?)', offset_str)
            if match:
                distance_val = float(match.group(1))
            elif offset_str in ('0', 'cl', 'centerline', 'center'):
                distance_val = 0.0
    if distance_val is not None:
        return (distance_val * direction, True) if distance_val != 0.0 else (0.0, True)
    return (None, False)

def find_value_from_keys(data_dict: dict, keys: list):
    for key in keys:
        if key in data_dict:
            return data_dict[key]
    return None

# --- COGO Functions ---

def prepare_centerline_data(centerline_geojson: dict) -> list:
    centerline_points = []
    for feature in centerline_geojson.get('features', []):
        props = feature.get('properties', {})
        geom = feature.get('geometry', {})
        if geom.get('type') != 'Point':
            continue
        station = parse_station_value_from_props(props)
        easting = props.get('calculated_grid_easting')
        northing = props.get('calculated_grid_northing')
        if station is not None and easting is not None and northing is not None:
            centerline_points.append((station, (easting, northing)))
    if not centerline_points:
        raise WorkflowError("No valid centerline points with stationing could be extracted.")
    centerline_points.sort(key=lambda p: p[0])
    print(f"  > Successfully prepared and sorted {len(centerline_points)} centerline points by station.")
    return centerline_points

def find_point_from_station_offset(station: float, grid_offset: float, sorted_centerline: list) -> tuple:
    if not sorted_centerline:
        raise WorkflowError("Centerline is empty, cannot calculate offset.")
    if station < sorted_centerline[0][0] or station > sorted_centerline[-1][0]:
        pass  # Silently extrapolate (v6 printed warnings, too noisy for v7 with many features)
    p1_station, p1_coords = sorted_centerline[0]
    p2_station, p2_coords = sorted_centerline[-1]
    for i in range(len(sorted_centerline) - 1):
        if sorted_centerline[i][0] <= station <= sorted_centerline[i + 1][0]:
            p1_station, p1_coords = sorted_centerline[i]
            p2_station, p2_coords = sorted_centerline[i + 1]
            break
    segment_vec_x = p2_coords[0] - p1_coords[0]
    segment_vec_y = p2_coords[1] - p1_coords[1]
    station_diff = p2_station - p1_station
    interp_ratio = (station - p1_station) / station_diff if station_diff != 0 else 0
    cl_easting = p1_coords[0] + interp_ratio * segment_vec_x
    cl_northing = p1_coords[1] + interp_ratio * segment_vec_y
    segment_length = math.hypot(segment_vec_x, segment_vec_y)
    if segment_length == 0:
        return (cl_easting, cl_northing)
    perp_vec_x = segment_vec_y / segment_length
    perp_vec_y = -segment_vec_x / segment_length
    return (cl_easting + grid_offset * perp_vec_x, cl_northing + grid_offset * perp_vec_y)

def generate_linestring_coords(
    start_station: float,
    end_station: float,
    offset_ft: float,
    sorted_centerline: list,
    combined_factor: float,
    transformer,
    point_spacing_ft: float = 25.0
) -> List[List[float]]:
    """Generate a series of WGS84 [lon, lat] coordinates for a line parallel to the centerline."""
    grid_offset = offset_ft / combined_factor
    coords = []
    station = start_station
    while station <= end_station:
        grid_e, grid_n = find_point_from_station_offset(station, grid_offset, sorted_centerline)
        lon, lat = transformer.transform(grid_e, grid_n)
        coords.append([lon, lat])
        station += point_spacing_ft
    # Always include the exact end station
    if not coords or station - point_spacing_ft < end_station:
        grid_e, grid_n = find_point_from_station_offset(end_station, grid_offset, sorted_centerline)
        lon, lat = transformer.transform(grid_e, grid_n)
        coords.append([lon, lat])
    return coords

def generate_polygon_coords(
    start_station: float,
    end_station: float,
    offset_left_ft: float,
    offset_right_ft: float,
    sorted_centerline: list,
    combined_factor: float,
    transformer,
    point_spacing_ft: float = 25.0
) -> List[List[List[float]]]:
    """Generate a closed polygon between two offset lines. Returns GeoJSON Polygon coordinates."""
    left_line = generate_linestring_coords(
        start_station, end_station, offset_left_ft,
        sorted_centerline, combined_factor, transformer, point_spacing_ft
    )
    right_line = generate_linestring_coords(
        start_station, end_station, offset_right_ft,
        sorted_centerline, combined_factor, transformer, point_spacing_ft
    )
    # Close the polygon: left forward, right reversed, close back to start
    ring = left_line + list(reversed(right_line))
    if ring:
        ring.append(ring[0])  # Close the ring
    return [ring]

def resolve_default_offset(item: dict, station: float, typical_sections: Dict) -> Optional[float]:
    """Assign a default offset based on typical section data when offset is missing."""
    if not typical_sections or 'sections' not in typical_sections:
        return None
    category = item.get('category', 'other')
    # Find the applicable typical section for this station
    section = typical_sections['sections'][0]  # Default to first section
    for s in typical_sections.get('sections', []):
        from_sta = parse_station_string(s.get('applies_from_station'))
        to_sta = parse_station_string(s.get('applies_to_station'))
        if from_sta is not None and to_sta is not None:
            if from_sta <= station <= to_sta:
                section = s
                break

    side_str = str(item.get('side', '')).lower()
    direction = -1 if 'lt' in side_str or 'left' in side_str else 1

    # Assign defaults based on category and typical section dimensions
    half_road = section.get('curb_to_curb_ft', 24) / 2
    shoulder = section.get('shoulder_right_ft', 4) if direction > 0 else section.get('shoulder_left_ft', 4)

    defaults = {
        'signing': half_road + shoulder + 6,
        'guardrail': half_road + shoulder + 2,
        'erosion_control': half_road + shoulder + 10,
        'traffic_control': half_road + shoulder + 6,
        'approach': half_road + shoulder,
        'curb_gutter': half_road,
        'sidewalk': half_road + shoulder + 1,
    }
    offset = defaults.get(category, half_road + shoulder + 5)
    return offset * direction

# --- Phase 1: Enhanced Metadata Extraction ---

def phase1_extract_metadata(pdf_path: str) -> Dict[str, Any]:
    print("\n--- Phase 1: Extracting Project Metadata ---")
    prompt = (
        "From the title sheet and survey data sheets of this civil engineering plan set, extract the "
        "following metadata. Return a JSON object with EXACTLY these keys:\n"
        "{\n"
        '  "project_name": "<project name/title from title sheet>",\n'
        '  "route": "<route designation, e.g. SH-27, US-93>",\n'
        '  "source_crs_epsg": <integer EPSG code: 2241 for Idaho East, 2242 for Idaho Central, 2243 for Idaho West>,\n'
        '  "combined_factor": <decimal number, the combined scale factor, typically near 1.0>,\n'
        '  "station_equations": [<array of station equation objects, or empty array []>],\n'
        '  "begin_station": "<beginning project station, e.g. 116+21>",\n'
        '  "end_station": "<ending project station, e.g. 256+15>"\n'
        "}\n\n"
        "IMPORTANT: The combined_factor converts ground distances to grid distances. It is typically "
        "a number very close to 1.0 (like 1.0002641834). Do NOT confuse it with the plan scale ratio. "
        "Use exactly these key names. Do not nest them inside another object."
    )
    metadata = call_gemini_api(pdf_path, prompt, is_json=True)
    print(f"  > Raw API response: {metadata}")

    # Fallback key resolution
    if "source_crs_epsg" not in metadata:
        for alt in ["epsg", "epsg_code", "crs_epsg", "EPSG", "spc_epsg", "state_plane_epsg"]:
            if alt in metadata:
                metadata["source_crs_epsg"] = metadata[alt]
                break
    if "combined_factor" not in metadata:
        for alt in ["scale_factor", "combined_scale_factor", "CSF", "grid_factor", "combinedFactor"]:
            if alt in metadata:
                metadata["combined_factor"] = metadata[alt]
                break

    if not all(k in metadata for k in ["source_crs_epsg", "combined_factor"]):
        raise WorkflowError("Metadata extraction failed. Response missing 'source_crs_epsg' or 'combined_factor'.")

    # Validate combined factor range
    cf = metadata['combined_factor']
    if isinstance(cf, (int, float)) and (cf < 0.999 or cf > 1.001):
        print(f"  > WARNING: Combined factor {cf} seems unusual. Typical range is 0.9995-1.0005.")

    print(f"  > Metadata extracted successfully:")
    print(f"    - Project: {metadata.get('project_name', 'N/A')}")
    print(f"    - Route: {metadata.get('route', 'N/A')}")
    print(f"    - EPSG: {metadata.get('source_crs_epsg')}")
    print(f"    - Combined Factor: {metadata.get('combined_factor')}")
    print(f"    - Station Range: {metadata.get('begin_station', '?')} to {metadata.get('end_station', '?')}")
    return metadata

# --- Phase 2: Typical Section Extraction ---

def phase2_extract_typical_section(pdf_path: str) -> Dict[str, Any]:
    print("\n--- Phase 2: Extracting Typical Section Data ---")
    prompt = (
        "From the Typical Section sheets of this plan set, extract the cross-section dimensions. "
        "Return a JSON object:\n"
        "{\n"
        '  "sections": [\n'
        "    {\n"
        '      "label": "<section label, e.g. Rural Section or Urban Section>",\n'
        '      "applies_from_station": "<start station or null>",\n'
        '      "applies_to_station": "<end station or null>",\n'
        '      "travel_lane_width_ft": <width of one travel lane in feet>,\n'
        '      "number_of_lanes": <total number of travel lanes>,\n'
        '      "shoulder_left_ft": <left shoulder width>,\n'
        '      "shoulder_right_ft": <right shoulder width>,\n'
        '      "curb_to_curb_ft": <total curb-to-curb or edge-to-edge width>,\n'
        '      "sidewalk_width_ft": <sidewalk width, 0 if none>,\n'
        '      "right_of_way_offset_left_ft": <ROW distance from centerline, left>,\n'
        '      "right_of_way_offset_right_ft": <ROW distance from centerline, right>\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Extract ALL typical sections shown. Each road segment may have a different cross-section."
    )
    try:
        typical = call_gemini_api(pdf_path, prompt, is_json=True)
        if 'sections' in typical and isinstance(typical['sections'], list):
            print(f"  > Extracted {len(typical['sections'])} typical section(s).")
            for s in typical['sections']:
                print(f"    - {s.get('label', 'Unnamed')}: {s.get('curb_to_curb_ft', '?')}ft curb-to-curb")
            return typical
    except WorkflowError:
        pass
    print("  > Could not extract typical section data. Using defaults.")
    return {"sections": [{"label": "Default", "curb_to_curb_ft": 24, "shoulder_left_ft": 4, "shoulder_right_ft": 4}]}

# --- Phase 3: Centerline Generation ---

def phase3_generate_centerline(pdf_path: str, metadata: Dict[str, Any]) -> Tuple[Dict[str, Any], list]:
    print("\n--- Phase 3: Generating Centerline Reference ---")
    print("  > Step 3.1: Extracting alignment control points as GeoJSON...")
    prompt = (
        "From the 'Survey Control/Monument Perpetuation' sheets or similar alignment data tables, "
        "extract the points that define the mainline alignment geometry. "
        "These are the points that have both stationing and coordinate values. "
        "The output must be a GeoJSON FeatureCollection. Each feature's properties must include "
        "its station, and its geometry coordinates must be the ground coordinates (Easting, Northing)."
    )
    ground_geojson = call_gemini_api(pdf_path, prompt, is_json=True)

    print("  > Step 3.2: Performing geodetic conversion...")
    source_crs_epsg = int(metadata['source_crs_epsg'])
    combined_factor = metadata['combined_factor']
    target_crs_epsg = 4326
    if not combined_factor or combined_factor == 0:
        raise WorkflowError("Combined factor is invalid.")

    source_crs = CRS.from_epsg(source_crs_epsg)
    target_crs = CRS.from_epsg(target_crs_epsg)
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

    processed_count = 0
    for feature in ground_geojson.get('features', []):
        geometry = feature.get('geometry')
        if not geometry or geometry.get('type') != 'Point':
            continue
        coords_ground = geometry.get('coordinates')
        if not isinstance(coords_ground, list) or len(coords_ground) < 2:
            continue
        easting_ground, northing_ground = coords_ground[0], coords_ground[1]
        easting_grid = easting_ground / combined_factor
        northing_grid = northing_ground / combined_factor
        lon, lat = transformer.transform(easting_grid, northing_grid)
        geometry['coordinates'] = [lon, lat]
        if 'properties' not in feature or not isinstance(feature['properties'], dict):
            feature['properties'] = {}
        feature['properties']['original_ground_easting'] = easting_ground
        feature['properties']['original_ground_northing'] = northing_ground
        feature['properties']['calculated_grid_easting'] = easting_grid
        feature['properties']['calculated_grid_northing'] = northing_grid
        processed_count += 1

    print(f"  > Converted {processed_count} control points to WGS84.")

    # Build the sorted centerline data structure
    sorted_centerline = prepare_centerline_data(ground_geojson)

    # Create a LineString feature connecting all control points in station order
    print("  > Step 3.3: Generating centerline LineString...")
    sorted_features = sorted(
        [f for f in ground_geojson['features'] if f.get('geometry', {}).get('type') == 'Point'],
        key=lambda f: parse_station_value_from_props(f.get('properties', {})) or 0
    )
    linestring_coords = [f['geometry']['coordinates'] for f in sorted_features]

    if len(linestring_coords) >= 2:
        centerline_line_feature = {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": linestring_coords},
            "properties": {
                "name": "Project Centerline",
                "description": f"{metadata.get('route', 'Alignment')} Centerline",
                "category": "centerline"
            }
        }
        ground_geojson['features'].append(centerline_line_feature)
        print(f"  > Centerline LineString generated with {len(linestring_coords)} vertices.")
    else:
        print("  > WARNING: Not enough points to generate centerline LineString.")

    return ground_geojson, sorted_centerline

# --- Phase 4: Sheet Inventory ---

def phase4_inventory_sheets(pdf_path: str) -> List[Dict]:
    print("\n--- Phase 4: Classifying Plan Set Sheets ---")
    prompt = (
        "Analyze this plan set PDF and create an inventory of ALL sheets. "
        "Return a JSON array where each entry represents a sheet or group of related sheets:\n\n"
        "[\n"
        "  {\n"
        '    "sheet_numbers": [1, 2],\n'
        '    "sheet_type": "Title & Index",\n'
        '    "description": "Title sheet and sheet index"\n'
        "  }\n"
        "]\n\n"
        "Use these standard sheet_type values: "
        '"Title & Index", "Summary of Quantities", "Typical Section", "Plan & Profile", '
        '"Drainage", "Signing", "Sign Summary", "Pavement Marking", "Striping", '
        '"Erosion Control", "SWPPP", "Traffic Control", "Survey Control", '
        '"Cross Section", "Utility", "Right of Way", "Bridge/Structure", '
        '"Landscaping", "Other"'
    )
    try:
        inventory = call_gemini_api(pdf_path, prompt, is_json=True)
        if isinstance(inventory, list):
            print(f"  > Identified {len(inventory)} sheet group(s):")
            for group in inventory:
                sheets = group.get('sheet_numbers', [])
                stype = group.get('sheet_type', 'Unknown')
                print(f"    - {stype}: sheet(s) {sheets}")
            return inventory
    except WorkflowError:
        pass
    print("  > Could not classify sheets. Will attempt all extraction types.")
    return [{"sheet_type": st, "title": st, "sheet_numbers": []} for st in
            ["Plan & Profile", "Roadway Summary", "Pipe Summary", "Drainage",
             "Signing", "Pavement Marking", "Erosion Control", "Traffic Control", "Utility"]]

# --- Phase 5: Category-Specific Feature Extraction (Parallel) ---

def _config_matches_inventory(config: dict, sheet_inventory: List[Dict]) -> bool:
    """Check if any keyword in the config matches any sheet title in the inventory (case-insensitive)."""
    all_titles = ' '.join(
        s.get('sheet_type', '') + ' ' + s.get('title', '') + ' ' + s.get('description', '')
        for s in sheet_inventory
    ).lower()
    return any(kw.lower() in all_titles for kw in config['match_keywords'])

def _extract_one_config(pdf_path: str, config: dict) -> Tuple[str, List[Dict]]:
    """Extract features for a single config. Returns (config_name, features_list)."""
    try:
        features = call_gemini_api(pdf_path, config['prompt'], is_json=True)
        if isinstance(features, list):
            for f in features:
                f['_extraction_source'] = config['name']
            return (config['name'], features)
        elif isinstance(features, dict) and 'features' in features:
            feature_list = features['features']
            for f in feature_list:
                f['_extraction_source'] = config['name']
            return (config['name'], feature_list)
        elif isinstance(features, dict) and isinstance(features.get('items'), list):
            feature_list = features['items']
            for f in feature_list:
                f['_extraction_source'] = config['name']
            return (config['name'], feature_list)
        else:
            return (config['name'], [])
    except WorkflowError as e:
        print(f"    ERROR extracting {config['name']}: {e}")
        return (config['name'], [])

def phase5_extract_all_features(pdf_path: str, sheet_inventory: List[Dict]) -> List[Dict]:
    print("\n--- Phase 5: Extracting Features by Category (Parallel) ---")

    # Determine which configs to run based on fuzzy keyword matching
    configs_to_run = []
    for config in EXTRACTION_CONFIGS:
        if _config_matches_inventory(config, sheet_inventory):
            configs_to_run.append(config)
            print(f"  > Queued: {config['name']}")
        else:
            print(f"  > Skipping '{config['name']}': no matching sheets found.")

    if not configs_to_run:
        print("  > WARNING: No extraction configs matched. Running all as fallback.")
        configs_to_run = EXTRACTION_CONFIGS

    # Run all extractions in parallel
    all_features = []
    print(f"\n  > Launching {len(configs_to_run)} parallel extraction(s)...")
    with ThreadPoolExecutor(max_workers=len(configs_to_run)) as executor:
        futures = {
            executor.submit(_extract_one_config, pdf_path, config): config['name']
            for config in configs_to_run
        }
        for future in as_completed(futures):
            config_name = futures[future]
            try:
                name, features = future.result()
                all_features.extend(features)
                print(f"    {name}: {len(features)} features")
            except Exception as e:
                print(f"    {config_name}: FAILED - {e}")

    print(f"\n  > Total raw features extracted across all categories: {len(all_features)}")
    return all_features

# --- Phase 6: Geometry Construction ---

def phase6_build_geometries(
    raw_features: List[Dict],
    sorted_centerline: list,
    metadata: Dict[str, Any],
    typical_sections: Dict[str, Any]
) -> Dict[str, Any]:
    print("\n--- Phase 6: Building Geometries (Point / LineString / Polygon) ---")
    combined_factor = metadata['combined_factor']
    source_crs_epsg = int(metadata['source_crs_epsg'])
    transformer = Transformer.from_crs(
        CRS.from_epsg(source_crs_epsg), CRS.from_epsg(4326), always_xy=True
    )

    output_features = []
    stats = {"point": 0, "line": 0, "polygon": 0, "skipped": 0}
    STATION_KEYS = ["start_station", "station", "centerline_distance", "sta"]

    for i, item in enumerate(raw_features):
        # Parse geometry type
        geom_type_raw = str(item.get('geometry_type', 'point')).lower().strip()
        if geom_type_raw in ('line', 'linestring', 'linear'):
            geom_type = 'line'
        elif geom_type_raw in ('polygon', 'poly', 'area'):
            geom_type = 'polygon'
        else:
            geom_type = 'point'

        # Parse stations
        raw_start = find_value_from_keys(item, STATION_KEYS)
        start_sta = parse_station_string(raw_start)
        end_sta = parse_station_string(item.get('end_station'))

        # Auto-detect line if end_station differs from start_station
        if start_sta is not None and end_sta is not None and end_sta > start_sta and geom_type == 'point':
            geom_type = 'line'

        # Parse offset
        ground_offset, offset_ok = parse_offset_value(item)

        if start_sta is None:
            desc = item.get('name', f'Item {i + 1}')
            print(f"    - Skipping '{desc}': missing station.")
            stats['skipped'] += 1
            continue

        # Resolve missing offset from typical section
        if not offset_ok:
            resolved = resolve_default_offset(item, start_sta, typical_sections)
            if resolved is not None:
                ground_offset = resolved
                offset_ok = True
                item['notes'] = item.get('notes', '') + ' [offset auto-assigned from typical section]'
            else:
                ground_offset = 0.0
                offset_ok = True
                item['notes'] = item.get('notes', '') + ' [offset defaulted to centerline]'

        category = item.get('category', 'other')

        try:
            if geom_type == 'line' and end_sta is not None and end_sta > start_sta:
                coords = generate_linestring_coords(
                    start_sta, end_sta, ground_offset,
                    sorted_centerline, combined_factor, transformer
                )
                if len(coords) >= 2:
                    geometry = {"type": "LineString", "coordinates": coords}
                    stats['line'] += 1
                else:
                    grid_offset = ground_offset / combined_factor
                    grid_e, grid_n = find_point_from_station_offset(start_sta, grid_offset, sorted_centerline)
                    lon, lat = transformer.transform(grid_e, grid_n)
                    geometry = {"type": "Point", "coordinates": [lon, lat]}
                    stats['point'] += 1

            elif geom_type == 'polygon' and end_sta is not None and end_sta > start_sta:
                # For polygons, create a band around the offset
                # Use a width of 10ft for approaches, 5ft for markings, etc.
                width_map = {
                    'approach': 12, 'pavement_marking': 4, 'roadway_surface': 0,
                    'sidewalk': 5, 'crosswalk': 10
                }
                half_width = width_map.get(category, 5)
                offset_left = ground_offset - half_width
                offset_right = ground_offset + half_width
                coords = generate_polygon_coords(
                    start_sta, end_sta, offset_left, offset_right,
                    sorted_centerline, combined_factor, transformer
                )
                geometry = {"type": "Polygon", "coordinates": coords}
                stats['polygon'] += 1

            else:
                # Point feature
                grid_offset = ground_offset / combined_factor
                grid_e, grid_n = find_point_from_station_offset(start_sta, grid_offset, sorted_centerline)
                lon, lat = transformer.transform(grid_e, grid_n)
                geometry = {"type": "Point", "coordinates": [lon, lat]}
                stats['point'] += 1

            feature = {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "name": item.get('name', ''),
                    "description": item.get('description', ''),
                    "category": category,
                    "start_station": item.get('start_station', str(start_sta)),
                    "end_station": item.get('end_station'),
                    "offset": item.get('offset', ''),
                    "side": item.get('side', ''),
                    "notes": item.get('notes', ''),
                    "parsed_start_station": start_sta,
                    "parsed_ground_offset_ft": ground_offset,
                    "_extraction_source": item.get('_extraction_source', ''),
                }
            }
            output_features.append(feature)

        except Exception as e:
            print(f"    - Error building geometry for '{item.get('name', '?')}' at sta {start_sta}: {e}")
            stats['skipped'] += 1

    print(f"\n  > Geometry construction complete:")
    print(f"    - Points:   {stats['point']}")
    print(f"    - Lines:    {stats['line']}")
    print(f"    - Polygons: {stats['polygon']}")
    print(f"    - Skipped:  {stats['skipped']}")

    return {
        "type": "FeatureCollection",
        "name": "Digital Twin Features",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::4326"}},
        "features": output_features
    }

# --- Phase 7: Output & Styled KMZ ---

def build_html_description(props: dict) -> str:
    desc = "<table>"
    skip_keys = {'_extraction_source', 'parsed_start_station', 'parsed_ground_offset_ft'}
    for key, value in props.items():
        if key in skip_keys or value is None or value == '':
            continue
        desc += f"<tr><td><b>{key}:</b></td><td>{value}</td></tr>"
    desc += "</table>"
    return desc

def create_styled_kmz(features_geojson: Dict, centerline_geojson: Dict, metadata: Dict, filename: str):
    """Create a KMZ file with folders per category and styled geometries."""
    project_name = metadata.get('project_name', 'Digital Twin')
    kml = simplekml.Kml(name=project_name)

    # Centerline folder
    cl_folder = kml.newfolder(name="Centerline")
    for f in centerline_geojson.get('features', []):
        geom = f.get('geometry', {})
        if geom.get('type') == 'LineString':
            ls = cl_folder.newlinestring(name="Project Centerline")
            ls.coords = [(c[0], c[1]) for c in geom['coordinates']]
            ls.style.linestyle.color = simplekml.Color.red
            ls.style.linestyle.width = 3
            ls.description = f"{metadata.get('route', '')} Centerline"
        elif geom.get('type') == 'Point':
            props = f.get('properties', {})
            pnt = cl_folder.newpoint(name=props.get('description', props.get('station', '')))
            pnt.coords = [(geom['coordinates'][0], geom['coordinates'][1])]
            pnt.style.iconstyle.color = simplekml.Color.red
            pnt.style.iconstyle.scale = 0.7
            pnt.description = build_html_description(props)

    # Create folders per category
    category_folders = {}
    for cat_name in CATEGORIES:
        if cat_name == 'centerline':
            continue
        folder = kml.newfolder(name=cat_name.replace('_', ' ').title())
        category_folders[cat_name] = folder

    feature_count = 0
    for feature in features_geojson.get('features', []):
        geom = feature.get('geometry', {})
        props = feature.get('properties', {})
        category = props.get('category', 'other')
        folder = category_folders.get(category, category_folders.get('other', kml))
        cat_config = CATEGORIES.get(category, CATEGORIES['other'])
        kml_color = cat_config.get('kml_color', 'ff696969')
        line_width = cat_config.get('width', 2)

        geom_type = geom.get('type')
        coords = geom.get('coordinates')
        if not coords:
            continue

        name = props.get('description') or props.get('name', f'Feature')
        html_desc = build_html_description(props)

        if geom_type == 'Point':
            pnt = folder.newpoint(name=name)
            pnt.coords = [(coords[0], coords[1])]
            pnt.style.iconstyle.color = kml_color
            pnt.style.iconstyle.scale = 0.8
            pnt.description = html_desc
            feature_count += 1

        elif geom_type == 'LineString':
            ls = folder.newlinestring(name=name)
            ls.coords = [(c[0], c[1]) for c in coords]
            ls.style.linestyle.color = kml_color
            ls.style.linestyle.width = line_width
            ls.description = html_desc
            feature_count += 1

        elif geom_type == 'Polygon':
            pol = folder.newpolygon(name=name)
            pol.outerboundaryis = [(c[0], c[1]) for c in coords[0]]
            # Semi-transparent fill
            pol.style.polystyle.color = simplekml.Color.changealphaint(80, int(kml_color, 16))
            pol.style.linestyle.color = kml_color
            pol.style.linestyle.width = 2
            pol.description = html_desc
            feature_count += 1

    print(f"  > KMZ file created with {feature_count} features across {len(category_folders) + 1} folders.")
    kml.savekmz(filename)

def phase7_save_outputs(centerline_geojson: Dict, features_geojson: Dict, metadata: Dict, pdf_path: str):
    print("\n--- Phase 7: Saving Outputs ---")

    # Build unique output prefix from PDF name + timestamp
    pdf_base = os.path.splitext(os.path.basename(pdf_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{pdf_base}_{timestamp}"

    # Create an output directory for this run
    output_dir = os.path.join("outputs", prefix)
    os.makedirs(output_dir, exist_ok=True)

    centerline_filename = os.path.join(output_dir, f"{prefix}_centerline.geojson")
    features_filename = os.path.join(output_dir, f"{prefix}_digital_twin.geojson")
    kmz_filename = os.path.join(output_dir, f"{prefix}_digital_twin.kmz")

    try:
        with open(centerline_filename, 'w') as f:
            json.dump(centerline_geojson, f, indent=2)
        print(f"  > Saved centerline to: '{centerline_filename}'")

        with open(features_filename, 'w') as f:
            json.dump(features_geojson, f, indent=2)
        print(f"  > Saved digital twin features to: '{features_filename}'")

        create_styled_kmz(features_geojson, centerline_geojson, metadata, kmz_filename)
        print(f"  > Saved styled KMZ to: '{kmz_filename}'")

        # Summary stats
        cats = {}
        for feat in features_geojson.get('features', []):
            cat = feat.get('properties', {}).get('category', 'other')
            gtype = feat.get('geometry', {}).get('type', 'Unknown')
            key = f"{cat}/{gtype}"
            cats[key] = cats.get(key, 0) + 1

        print("\n--- Summary ---")
        print(f"  Total features: {len(features_geojson.get('features', []))}")
        for key in sorted(cats.keys()):
            print(f"    {key}: {cats[key]}")

        print("\n--- Validation ---")
        print("Process complete. Validate the results:")
        print(f"  1. Open '{kmz_filename}' in Google Earth Pro (recommended)")
        print(f"  2. Drag '{features_filename}' into https://geojson.io/")

    except Exception as e:
        raise WorkflowError(f"Failed to write output files: {e}")

# --- Main Execution ---

def main():
    print("=" * 55)
    print(" Digital Twin: Automated PDF Plan Set Processing (v7)")
    print("=" * 55)

    if len(sys.argv) < 2:
        print(f"\nUsage: python {sys.argv[0]} <path_to_your_pdf_file.pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"\nError: The file '{pdf_path}' was not found.")
        sys.exit(1)

    try:
        api_key = get_api_key()
        genai.configure(api_key=api_key)

        # Phase 1: Metadata
        project_metadata = phase1_extract_metadata(pdf_path)

        print("\n--- User Validation Required ---")
        print(f"  > AI extracted the following metadata:")
        print(f"    - Project: {project_metadata.get('project_name', 'N/A')}")
        print(f"    - Route: {project_metadata.get('route', 'N/A')}")
        print(f"    - EPSG Code: {project_metadata.get('source_crs_epsg')}")
        print(f"    - Combined Factor: {project_metadata.get('combined_factor')}")
        confirm = input("  > Does this information look correct? (yes/no): ").lower().strip()
        if confirm not in ['y', 'yes']:
            print("  > Aborting workflow based on user input.")
            sys.exit(1)

        # Phase 2 + Phase 4: Run in parallel (independent of each other)
        print("\n  > Running Phase 2 (Typical Section) and Phase 4 (Sheet Inventory) in parallel...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_typical = executor.submit(phase2_extract_typical_section, pdf_path)
            future_inventory = executor.submit(phase4_inventory_sheets, pdf_path)
            typical_sections = future_typical.result()
            sheet_inventory = future_inventory.result()

        # Phase 3: Centerline (depends on Phase 1 metadata)
        centerline_geojson, sorted_centerline = phase3_generate_centerline(pdf_path, project_metadata)

        # Phase 5: Category-Specific Feature Extraction
        raw_features = phase5_extract_all_features(pdf_path, sheet_inventory)

        # Phase 6: Geometry Construction
        features_geojson = phase6_build_geometries(
            raw_features, sorted_centerline, project_metadata, typical_sections
        )

        # Phase 7: Output
        phase7_save_outputs(centerline_geojson, features_geojson, project_metadata, pdf_path)

    except WorkflowError as e:
        print(f"\nA workflow error occurred: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
