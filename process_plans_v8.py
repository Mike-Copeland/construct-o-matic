# process_plans_v8.py — Project Spatial Model Builder
# Builds a precise 3D alignment model from plan set PDFs.
# Outputs: spatial_model.json (for downstream tools) + styled KMZ (for HITL validation)
# Core: AlignmentEngine with true arc geometry from horizontal PI data + vertical profile from VPI data.

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
from alignment_engine import AlignmentEngine, parse_station, parse_numeric, parse_dms_angle

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
#
# ITD Plan Set Structure Reference:
#   Plan sheets show a PLAN VIEW (top) and PROFILE VIEW (bottom).
#   The plan view is a scaled drawing with station tick marks along the centerline.
#   Features are drawn at their physical location with callout labels showing item codes.
#   Station+offset can be read from the drawing by referencing the station tick marks.
#   Summary/detail sheets use ITD standard spec numbering:
#     1xx=General, 2xx=Earthwork/Erosion, 3xx=Base, 4xx=Surfacing,
#     5xx=Structures, 6xx=Roadside (60x=Drainage, 61x=Signing, 63x=Markings)

FEATURE_SCHEMA_BLOCK = (
    'Return a JSON array. Each object MUST have these keys:\n'
    '{"name":"<item code>","description":"<description>","category":"<see below>",'
    '"geometry_type":"<point or line>","start_station":"<e.g. 116+20>",'
    '"end_station":"<end station or null>","offset":"<e.g. 35\' Lt.>",'
    '"side":"<Lt or Rt or CL>","notes":"<additional info>"}\n'
)

PLAN_VIEW_CONTEXT = (
    "\n\nHOW TO READ THESE SHEETS:\n"
    "These are civil engineering plan sheets. The PLAN VIEW is a scaled aerial/top-down drawing. "
    "Station tick marks are shown along the centerline at regular intervals (typically every 100 feet, "
    "formatted as e.g. 125+00, 126+00). Features are drawn at their physical location on the plan view. "
    "To determine a feature's station: find the nearest station tick marks on the centerline and "
    "interpolate. To determine offset: estimate the perpendicular distance from centerline, noting "
    "whether the feature is left (Lt) or right (Rt) of centerline when looking ahead (in the direction "
    "of increasing stations). Read callout labels, leader lines, and annotations for item codes and "
    "descriptions. Extract data from BOTH the drawings AND any tables on the sheets.\n"
)

EXTRACTION_CONFIGS = [
    {
        "name": "Roadway Summary",
        "match_keywords": ["roadway summary", "summary of quantities"],
        "prompt": (
            "From the ROADWAY SUMMARY / SUMMARY OF QUANTITIES sheets, extract ALL pay items that have "
            "station references. These tabular sheets list construction items with quantities.\n\n"
            "ONLY extract items that have a specific station or station range listed in the table. "
            "Skip lump-sum items that have no station reference — they cannot be geolocated.\n\n"
            "For items with a station range (e.g. Sta 125+00 to 283+48), set geometry_type to 'line'.\n"
            "For items at a single station, set geometry_type to 'point'.\n"
            "Set category to the best fit: curb_gutter, sidewalk, guardrail, drainage, "
            "pavement_marking, erosion_control, signing, approach, roadway_surface, or other.\n\n"
            + FEATURE_SCHEMA_BLOCK
        ),
    },
    {
        "name": "Pipe & Culvert Summary",
        "match_keywords": ["pipe", "culvert"],
        "prompt": (
            "From the PIPE CULVERT SUMMARY or PIPE SUMMARY table, extract ALL pipes and culverts.\n\n"
            "These tables list each pipe/culvert with its station, size, length, material, and type. "
            "Each pipe is a POINT feature at its centerline station. "
            "Cross-reference the plan sheets if a pipe's station or offset is ambiguous.\n\n"
            "Set category to 'drainage' for all items.\n\n"
            + FEATURE_SCHEMA_BLOCK +
            "Include pipe diameter, material, length, and slope in the notes field."
        ),
    },
    {
        "name": "Plan & Profile Features",
        "match_keywords": ["plan & profile", "plan sheet", "plan/profile", "plan/ profile"],
        "prompt": (
            "From the PLAN & PROFILE sheets, extract ALL constructed features shown in the PLAN VIEW drawings. "
            "The plan view is the top portion of each sheet — a scaled top-down drawing of the road.\n\n"
            "Read the drawings carefully. Features include:\n"
            "- Curb & gutter runs (drawn as lines along the road edge)\n"
            "- Sidewalk (drawn parallel to curb)\n"
            "- Guardrail / barrier (drawn with standard symbols)\n"
            "- Approaches / driveways (connections to adjacent properties)\n"
            "- Retaining walls, headwalls, end sections\n"
            "- Fence lines (drawn with X-pattern or specific line type)\n"
            "- Ditches and channels (drawn with standard symbols)\n"
            "- Any feature with an item code callout (e.g. 405-245A, 615-493A)\n\n"
            "DETERMINE STATION by reading the station tick marks along the centerline "
            "(marked at 100-ft intervals, e.g. 125+00, 126+00). Interpolate between tick marks.\n"
            "DETERMINE OFFSET by estimating the perpendicular distance from centerline.\n\n"
            "LINEAR features (curb, sidewalk, guardrail, fence, ditch) MUST have both "
            "start_station and end_station.\n"
            "POINT features (approaches, isolated structures) need only start_station.\n\n"
            "Set category to: curb_gutter, sidewalk, guardrail, approach, drainage, or other.\n\n"
            + FEATURE_SCHEMA_BLOCK +
            PLAN_VIEW_CONTEXT +
            "Extract EVERY visible feature. Do not skip any."
        ),
    },
    {
        "name": "Drainage Features",
        "match_keywords": ["drainage"],
        "prompt": (
            "From the DRAINAGE plan sheets, extract ALL drainage structures and features. "
            "These sheets show drainage infrastructure in plan view with callout labels.\n\n"
            "Read BOTH the plan view drawings AND any drainage structure tables/schedules.\n\n"
            "Features to extract:\n"
            "- Inlets (all types) — POINT at their station, with offset\n"
            "- Manholes — POINT\n"
            "- Catch basins — POINT\n"
            "- Pipe runs between structures — LINE from upstream to downstream station\n"
            "- Swales and ditches — LINE with start/end station\n"
            "- Retention/detention ponds — POINT at center\n"
            "- Culverts and cross drains — POINT at centerline station\n"
            "- Headwalls and end sections — POINT\n"
            "- Outfall structures — POINT\n\n"
            "Set category to 'drainage'.\n\n"
            + FEATURE_SCHEMA_BLOCK +
            PLAN_VIEW_CONTEXT +
            "Include pipe size, material, slope, and connected structure IDs in notes."
        ),
    },
    {
        "name": "Signing Features",
        "match_keywords": ["sign"],
        "prompt": (
            "Extract ALL signs from this plan set. Look in MULTIPLE locations:\n\n"
            "1. SIGN SUMMARY TABLE — lists sign types, MUTCD codes, and quantities. "
            "This table may or may not include station references.\n"
            "2. SIGN & PAVEMENT MARKING PLAN sheets — plan view drawings showing each sign's "
            "physical location along the road with station tick marks on the centerline.\n"
            "3. PLAN & PROFILE sheets — may show signs with callout labels.\n\n"
            "For signs in the summary table WITHOUT stations: look at the plan view sheets to find "
            "where that sign type appears and read the station from the drawing.\n\n"
            "Each sign is a POINT feature. Signs are typically offset 6-12 feet from the edge of "
            "the travel lane. Note the side (Lt/Rt) based on which side of the road the sign is drawn.\n\n"
            "Set category to 'signing'.\n\n"
            + FEATURE_SCHEMA_BLOCK +
            PLAN_VIEW_CONTEXT +
            "Include MUTCD code, sign legend/message, facing direction, "
            "and new/existing/relocate/remove status in notes."
        ),
    },
    {
        "name": "Pavement Marking Features",
        "match_keywords": ["pavement mark", "striping", "marking plan"],
        "prompt": (
            "From the PAVEMENT MARKING, STRIPING, and SIGN & PAVEMENT MARKING PLAN sheets, "
            "extract ALL pavement markings shown in the plan view drawings.\n\n"
            "These sheets show markings drawn on the road surface with labels and callouts.\n\n"
            "Features to extract:\n"
            "- Center line striping — LINE from start to end station, offset = 0' CL\n"
            "- Edge lines (white solid) — LINE, offset = half the road width Lt and Rt\n"
            "- Lane lines — LINE between travel lanes\n"
            "- Skip lines / broken lines — LINE with pattern noted\n"
            "- Turn arrows — POINT at their station\n"
            "- ONLY/STOP/YIELD word markings — POINT\n"
            "- Crosswalk markings — POINT or LINE\n"
            "- Stop bars — POINT\n"
            "- Railroad crossing markings — POINT\n\n"
            "DETERMINE STATION from the station tick marks on the centerline.\n"
            "Set category to 'pavement_marking'.\n\n"
            + FEATURE_SCHEMA_BLOCK +
            PLAN_VIEW_CONTEXT +
            "Include color (white/yellow), width (4\"/6\"/8\"/12\"/24\"), "
            "and pattern (solid/skip/broken/dotted) in notes."
        ),
    },
    {
        "name": "Erosion Control Features",
        "match_keywords": ["erosion", "swppp", "212-", "bmp"],
        "prompt": (
            "From the EROSION CONTROL, SWPPP, and BMP sheets (including ITD standard detail sheets "
            "numbered 212-xxx), extract ALL erosion control / stormwater BMPs.\n\n"
            "Also check the PLAN & PROFILE sheets for erosion control items drawn in the plan view "
            "(silt fence is often drawn as a dashed line with X marks, inlet protection shown at inlets).\n\n"
            "Features to extract:\n"
            "- Silt fence — LINE with start/end station, typically at construction limits\n"
            "- Inlet protection — POINT at each protected inlet\n"
            "- Erosion blanket / erosion mat — LINE over the covered area\n"
            "- Wattles / fiber rolls — LINE along contours\n"
            "- Seeding/mulching areas — LINE with start/end station if station range given\n"
            "- Check dams — POINT in ditches\n"
            "- Sediment traps/basins — POINT\n"
            "- Construction entrances — POINT\n"
            "- Concrete washout areas — POINT\n\n"
            "Set category to 'erosion_control'.\n\n"
            + FEATURE_SCHEMA_BLOCK +
            PLAN_VIEW_CONTEXT
        ),
    },
    {
        "name": "Traffic Control Features",
        "match_keywords": ["traffic control", "traffic plan", "tcp"],
        "prompt": (
            "From the TEMPORARY TRAFFIC CONTROL PLAN (TCP) sheets, extract ALL traffic control devices "
            "shown in the plan view drawings.\n\n"
            "TCP sheets are plan view drawings showing temporary signing, lane closures, "
            "and work zone layouts. They use the same station tick marks as the plan sheets.\n\n"
            "READ THE PLAN VIEW DRAWING to determine each device's station and offset. "
            "Do NOT rely solely on tables — the device locations are shown graphically.\n\n"
            "Features to extract:\n"
            "- Temporary signs (shown with MUTCD codes like W20-1, G20-1) — POINT\n"
            "- Advance warning signs — POINT at their drawn location\n"
            "- Channelizing devices / barricades — POINT or LINE if defining a taper\n"
            "- Arrow boards — POINT\n"
            "- Flagging stations — POINT where the flagger symbol is drawn\n"
            "- Temporary striping — LINE\n"
            "- Work zone limits — LINE from start to end station\n"
            "- Pilot car turnouts — POINT\n\n"
            "IMPORTANT: TCP plans often show multiple PHASES. Extract devices from ALL phases. "
            "Note which phase each device belongs to in the notes field.\n\n"
            "Set category to 'traffic_control'.\n\n"
            + FEATURE_SCHEMA_BLOCK +
            PLAN_VIEW_CONTEXT +
            "Include the TCP phase number and device type in notes."
        ),
    },
    {
        "name": "Utility Features",
        "match_keywords": ["utility"],
        "prompt": (
            "From the UTILITY PLANS sheets, extract ALL utility features shown in the plan view.\n\n"
            "These sheets show existing and proposed utility lines, poles, and structures "
            "drawn over the road plan view with station tick marks.\n\n"
            "Features to extract:\n"
            "- Utility poles (power, telecom) — POINT with station and offset\n"
            "- Underground utility crossings — POINT at centerline station\n"
            "- Utility relocations — LINE from old to new location, or POINT if single structure\n"
            "- Manholes / vaults / junction boxes — POINT\n"
            "- Conduit runs — LINE with start/end station\n"
            "- Overhead lines — LINE if station range is clear\n"
            "- Fire hydrants — POINT\n"
            "- Irrigation structures — POINT\n\n"
            "Set category to 'utility'.\n\n"
            + FEATURE_SCHEMA_BLOCK +
            PLAN_VIEW_CONTEXT +
            "Include utility owner (power company, telco, irrigation district, city) "
            "and type (electric, gas, water, telecom, irrigation, fiber) in notes."
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

# --- Offset Resolution ---

def resolve_default_offset(item: dict, station: float, engine: AlignmentEngine) -> Optional[float]:
    """Assign a default offset based on typical section data when offset is missing."""
    section = engine.get_section(station)
    category = item.get('category', 'other')

    side_str = str(item.get('side', '')).lower()
    direction = -1 if 'lt' in side_str or 'left' in side_str else 1

    half_road = section.curb_to_curb / 2
    shoulder = section.shoulder_right if direction > 0 else section.shoulder_left

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

# --- Phase 2: Alignment Geometry Extraction (replaces v7 Phase 2 + Phase 3) ---

ALIGNMENT_PROMPT = (
    "From the Survey Control / Monument Perpetuation sheets AND the Plan & Profile sheets, "
    "extract the COMPLETE alignment geometry. Return a JSON object with EXACTLY these sections:\n\n"
    "{\n"
    '  "horizontal_pis": [\n'
    "    {\n"
    '      "pi_station": "<e.g. 128+39.42>",\n'
    '      "delta": "<e.g. 21°27\'17\\" LT>",\n'
    '      "radius": <number in feet>,\n'
    '      "tangent_length": <number in feet>,\n'
    '      "curve_length": <number in feet>,\n'
    '      "pc_station": "<e.g. 123+94.23>",\n'
    '      "pt_station": "<e.g. 132+74.19>",\n'
    '      "northing": <ground coordinate>,\n'
    '      "easting": <ground coordinate>,\n'
    '      "superelevation": "<e.g. 5.2%>"\n'
    "    }\n"
    "  ],\n"
    '  "vertical_pis": [\n'
    "    {\n"
    '      "vpi_station": "<e.g. 133+55.00>",\n'
    '      "vpi_elevation": <number in feet>,\n'
    '      "curve_length": <vertical curve length in feet, 0 if no curve>,\n'
    '      "k_value": <K value>,\n'
    '      "vpc_station": "<start of vertical curve>",\n'
    '      "vpt_station": "<end of vertical curve>"\n'
    "    }\n"
    "  ],\n"
    '  "control_points": [\n'
    "    {\n"
    '      "point_number": "<id>",\n'
    '      "station": "<station>",\n'
    '      "offset": "<offset from CL>",\n'
    '      "northing": <ground coordinate>,\n'
    '      "easting": <ground coordinate>,\n'
    '      "elevation": <if available, else null>\n'
    "    }\n"
    "  ]\n"
    "}\n\n"
    "CRITICAL: Extract ALL horizontal PIs from the alignment data table (they have delta angles, "
    "radii, tangent lengths). Extract ALL vertical PIs from the profile view (they have VPI "
    "stations, elevations, and curve lengths). Extract ALL survey control/monument points with "
    "their coordinates. Do not skip any data."
)

TYPICAL_SECTION_PROMPT = (
    "From the Typical Section sheets, extract cross-section dimensions. Return a JSON object:\n"
    '{"sections": [{"label":"<name>","station_range":"<e.g. Sta 124+06 - 158+50>",'
    '"lane_widths":{"left":<ft>,"right":<ft>},"shoulder_widths":{"left":<ft>,"right":<ft>},'
    '"curb_to_curb_ft":<ft>,"sidewalk_width_ft":<ft>,'
    '"right_of_way_offset_left_ft":<ft>,"right_of_way_offset_right_ft":<ft>}]}'
)

def phase2_build_alignment(pdf_path: str, metadata: Dict[str, Any]) -> AlignmentEngine:
    """Extract alignment geometry and build the AlignmentEngine."""
    print("\n--- Phase 2: Building Alignment Engine ---")

    combined_factor = metadata['combined_factor']
    begin_sta = parse_station(metadata.get('begin_station')) or 0
    end_sta = parse_station(metadata.get('end_station')) or 99999

    # Run alignment extraction and typical section extraction in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_align = executor.submit(call_gemini_api, pdf_path, ALIGNMENT_PROMPT, True)
        future_typical = executor.submit(call_gemini_api, pdf_path, TYPICAL_SECTION_PROMPT, True)
        alignment_data = future_align.result()
        typical_data = future_typical.result()

    engine = AlignmentEngine()
    engine.combined_factor = combined_factor

    # Build horizontal alignment from PI data
    h_pis = alignment_data.get('horizontal_pis', [])
    print(f"  > Horizontal PIs extracted: {len(h_pis)}")
    if h_pis:
        engine.build_horizontal(h_pis, begin_sta, end_sta)
    else:
        # Fallback: build from control points using linear interpolation
        print("  > WARNING: No horizontal PI data. Falling back to control point interpolation.")
        control_pts = alignment_data.get('control_points', [])
        if control_pts:
            # Create pseudo-PIs from control points (no curves, tangent-only)
            engine.begin_station = begin_sta
            engine.end_station = end_sta

    # Build vertical profile from VPI data
    v_pis = alignment_data.get('vertical_pis', [])
    print(f"  > Vertical PIs extracted: {len(v_pis)}")
    if v_pis:
        engine.build_vertical(v_pis)
    else:
        print("  > No vertical PI data available. Elevation will not be computed.")

    # Build cross-section model
    sections = typical_data.get('sections', []) if isinstance(typical_data, dict) else []
    if sections:
        engine.build_cross_sections(sections)
    else:
        print("  > No typical section data. Using defaults.")
        engine.build_cross_sections([{"label": "Default"}])

    return engine

# --- Phase 3: Generate Centerline from AlignmentEngine ---

def phase3_generate_centerline(engine: AlignmentEngine, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a dense 3D centerline GeoJSON from the alignment engine."""
    print("\n--- Phase 3: Generating 3D Centerline from Alignment Engine ---")

    source_crs_epsg = int(metadata['source_crs_epsg'])
    combined_factor = metadata['combined_factor']
    transformer = Transformer.from_crs(CRS.from_epsg(source_crs_epsg), CRS.from_epsg(4326), always_xy=True)

    # Generate dense centerline coordinates (every 25ft, following true arcs)
    cl_coords = engine.generate_centerline_coords(combined_factor, transformer, spacing_ft=25.0)
    print(f"  > Centerline generated with {len(cl_coords)} vertices (25ft spacing, true arc geometry)")

    has_3d = any(len(c) == 3 for c in cl_coords)
    print(f"  > 3D elevation: {'Yes' if has_3d else 'No (2D only)'}")

    centerline_geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": cl_coords},
            "properties": {
                "name": "Project Centerline",
                "description": f"{metadata.get('route', 'Alignment')} Centerline",
                "category": "centerline",
                "begin_station": engine.begin_station,
                "end_station": engine.end_station,
                "num_horizontal_curves": len(engine.horizontal_pis),
                "num_vertical_curves": len([v for v in engine.vertical_pis if v.curve_length > 0]),
            }
        }]
    }

    return centerline_geojson

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
    """Check if any keyword in the config matches any field in the inventory (case-insensitive).
    Searches sheet_type, title, description, AND sheet_numbers."""
    parts = []
    for s in sheet_inventory:
        parts.append(s.get('sheet_type', ''))
        parts.append(s.get('title', ''))
        parts.append(s.get('description', ''))
        # Include sheet numbers — catches ITD spec references like "212-15" for erosion
        nums = s.get('sheet_numbers', [])
        if isinstance(nums, list):
            parts.extend(str(n) for n in nums)
    all_text = ' '.join(parts).lower()
    return any(kw.lower() in all_text for kw in config['match_keywords'])

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
    engine: AlignmentEngine,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    print("\n--- Phase 6: Building Geometries via AlignmentEngine ---")
    combined_factor = metadata['combined_factor']
    source_crs_epsg = int(metadata['source_crs_epsg'])
    transformer = Transformer.from_crs(
        CRS.from_epsg(source_crs_epsg), CRS.from_epsg(4326), always_xy=True
    )

    output_features = []
    stats = {"point": 0, "line": 0, "skipped": 0}
    STATION_KEYS = ["start_station", "station", "centerline_distance", "sta"]

    for i, item in enumerate(raw_features):
        geom_type_raw = str(item.get('geometry_type', 'point')).lower().strip()
        geom_type = 'line' if geom_type_raw in ('line', 'linestring', 'linear') else 'point'

        raw_start = find_value_from_keys(item, STATION_KEYS)
        start_sta = parse_station_string(raw_start)
        end_sta = parse_station_string(item.get('end_station'))

        if start_sta is not None and end_sta is not None and end_sta > start_sta and geom_type == 'point':
            geom_type = 'line'

        ground_offset, offset_ok = parse_offset_value(item)

        if start_sta is None:
            desc = item.get('name', f'Item {i + 1}')
            print(f"    - Skipping '{desc}': missing station.")
            stats['skipped'] += 1
            continue

        if not offset_ok:
            resolved = resolve_default_offset(item, start_sta, engine)
            if resolved is not None:
                ground_offset = resolved
                offset_ok = True
                item['notes'] = item.get('notes', '') + ' [offset auto-assigned]'
            else:
                ground_offset = 0.0
                offset_ok = True
                item['notes'] = item.get('notes', '') + ' [offset defaulted to CL]'

        category = item.get('category', 'other')
        elevation = engine.station_to_elevation(start_sta)

        try:
            if geom_type == 'line' and end_sta is not None and end_sta > start_sta:
                coords = engine.generate_linestring_coords(
                    start_sta, end_sta, ground_offset,
                    combined_factor, transformer
                )
                if len(coords) >= 2:
                    geometry = {"type": "LineString", "coordinates": coords}
                    stats['line'] += 1
                else:
                    ground_e, ground_n = engine.station_offset_to_grid_coords(start_sta, ground_offset)
                    lon, lat = transformer.transform(ground_e / combined_factor, ground_n / combined_factor)
                    coord = [lon, lat, elevation] if elevation else [lon, lat]
                    geometry = {"type": "Point", "coordinates": coord}
                    stats['point'] += 1
            else:
                ground_e, ground_n = engine.station_offset_to_grid_coords(start_sta, ground_offset)
                lon, lat = transformer.transform(ground_e / combined_factor, ground_n / combined_factor)
                coord = [lon, lat, elevation] if elevation else [lon, lat]
                geometry = {"type": "Point", "coordinates": coord}
                stats['point'] += 1

            feature = {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "name": item.get('name', ''),
                    "description": item.get('description', ''),
                    "category": category,
                    "station": start_sta,
                    "end_station": end_sta,
                    "offset_ft": ground_offset,
                    "side": item.get('side', ''),
                    "elevation": elevation,
                    "notes": item.get('notes', ''),
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

def phase7_save_outputs(centerline_geojson: Dict, features_geojson: Dict,
                        metadata: Dict, engine: AlignmentEngine, pdf_path: str):
    print("\n--- Phase 7: Saving Outputs ---")

    pdf_base = os.path.splitext(os.path.basename(pdf_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{pdf_base}_{timestamp}"

    output_dir = os.path.join("outputs", prefix)
    os.makedirs(output_dir, exist_ok=True)

    centerline_filename = os.path.join(output_dir, f"{prefix}_centerline.geojson")
    features_filename = os.path.join(output_dir, f"{prefix}_features.geojson")
    spatial_model_filename = os.path.join(output_dir, f"{prefix}_spatial_model.json")
    kmz_filename = os.path.join(output_dir, f"{prefix}.kmz")

    try:
        with open(centerline_filename, 'w') as f:
            json.dump(centerline_geojson, f, indent=2)
        print(f"  > Saved centerline to: '{centerline_filename}'")

        with open(features_filename, 'w') as f:
            json.dump(features_geojson, f, indent=2)
        print(f"  > Saved digital twin features to: '{features_filename}'")

        # Build and save the spatial model JSON (for downstream tool consumption)
        spatial_model = {
            "project": {
                "name": metadata.get('project_name', ''),
                "route": metadata.get('route', ''),
                "key_number": metadata.get('key_number', pdf_base),
                "epsg": int(metadata.get('source_crs_epsg', 0)),
                "combined_factor": metadata.get('combined_factor'),
                "begin_station": engine.begin_station,
                "end_station": engine.end_station,
                "begin_milepost": metadata.get('begin_milepost'),
                "end_milepost": metadata.get('end_milepost'),
            },
            "alignment": {
                "horizontal_pis": [
                    {"station": pi.station, "easting": pi.easting, "northing": pi.northing,
                     "delta_rad": pi.delta_rad, "radius": pi.radius,
                     "tangent_length": pi.tangent_length, "curve_length": pi.curve_length,
                     "pc_station": pi.pc_station, "pt_station": pi.pt_station}
                    for pi in engine.horizontal_pis
                ],
                "vertical_pis": [
                    {"station": vpi.station, "elevation": vpi.elevation,
                     "curve_length": vpi.curve_length, "k_value": vpi.k_value,
                     "vpc_station": vpi.vpc_station, "vpt_station": vpi.vpt_station}
                    for vpi in engine.vertical_pis
                ],
                "typical_sections": [
                    {"label": s.label, "start_station": s.start_station,
                     "end_station": s.end_station, "curb_to_curb": s.curb_to_curb,
                     "lane_width_left": s.lane_width_left, "lane_width_right": s.lane_width_right,
                     "shoulder_left": s.shoulder_left, "shoulder_right": s.shoulder_right,
                     "row_offset_left": s.row_offset_left, "row_offset_right": s.row_offset_right}
                    for s in engine.typical_sections
                ],
            },
            "features": features_geojson.get('features', []),
        }
        with open(spatial_model_filename, 'w') as f:
            json.dump(spatial_model, f, indent=2)
        print(f"  > Saved spatial model to: '{spatial_model_filename}'")

        create_styled_kmz(features_geojson, centerline_geojson, metadata, kmz_filename)
        print(f"  > Saved KMZ to: '{kmz_filename}'")

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
    print("=" * 60)
    print(" Project Spatial Model Builder (v8)")
    print(" True arc geometry | 3D elevation | Spatial model output")
    print("=" * 60)

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

        # Phase 2: Build Alignment Engine (horizontal curves + vertical profile + typical sections)
        # Runs alignment + typical section extraction in parallel internally
        engine = phase2_build_alignment(pdf_path, project_metadata)

        # Phase 3: Generate 3D centerline from alignment engine
        centerline_geojson = phase3_generate_centerline(engine, project_metadata)

        # Phase 4: Sheet Inventory (for driving Phase 5 extraction)
        sheet_inventory = phase4_inventory_sheets(pdf_path)

        # Phase 5: Category-Specific Feature Extraction (parallel)
        raw_features = phase5_extract_all_features(pdf_path, sheet_inventory)

        # Phase 6: Geometry Construction (using AlignmentEngine for true arc geometry)
        features_geojson = phase6_build_geometries(raw_features, engine, project_metadata)

        # Phase 7: Output — spatial model JSON + KMZ for HITL validation
        phase7_save_outputs(centerline_geojson, features_geojson, project_metadata, engine, pdf_path)

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
