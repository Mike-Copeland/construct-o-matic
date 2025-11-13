# process_plans.py (Version 10 - KMZ Export Added)

import google.generativeai as genai
import json
import os
import math
import re
import sys
from pyproj import CRS, Transformer
from typing import Optional, Tuple, List, Dict, Any
import simplekml # <-- ADDED FOR KMZ EXPORT

# --- Configuration ---
MODEL_NAME = "gemini-2.5-pro" 

# --- Helper Functions & Classes ---

class WorkflowError(Exception):
    """Custom exception for workflow-specific errors."""
    pass

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

def call_gemini_api(pdf_path: str, prompt: str, is_json: bool = True) -> Any:
    print(f"  > Calling Gemini API ('{MODEL_NAME}') for: '{prompt[:50]}...'")
    try:
        uploaded_file = genai.upload_file(path=pdf_path, display_name="Plan Set PDF")
        print(f"  > Successfully uploaded file: {uploaded_file.display_name}")
        model = genai.GenerativeModel(MODEL_NAME)
        generation_config = {"temperature": 0.0}
        if is_json:
            generation_config["response_mime_type"] = "application/json"
        response = model.generate_content([prompt, uploaded_file], generation_config=generation_config)
        cleaned_text = re.sub(r'^```json\s*|\s*```$', '', response.text.strip(), flags=re.MULTILINE)
        return json.loads(cleaned_text)
    except Exception as e:
        print(f"  > An error occurred during the API call: {e}")
        raise WorkflowError(f"Failed to get a valid response from the Gemini API.")

# --- Phase 1: Metadata Extraction ---

def phase1_extract_metadata(pdf_path: str) -> Dict[str, Any]:
    print("\n--- Phase 1: Extracting Project Metadata ---")
    prompt = (
        "From the survey or title sheets, give a json of the data needed to convert any point from the project level stationing to lat/long coordinates. We need: "
        "'source_crs_epsg' (e.g., 2241, 2242, or 2243 for Idaho), 'combined_factor', and any 'station_equations'."
    )
    metadata = call_gemini_api(pdf_path, prompt, is_json=True)
    if not all(k in metadata for k in ["source_crs_epsg", "combined_factor"]):
        raise WorkflowError("Metadata extraction failed. Response missing 'source_crs_epsg' or 'combined_factor'.")
    print(f"  > Metadata extracted successfully: {metadata}")
    return metadata

# --- Phase 2: Centerline Generation ---

def phase2_generate_centerline(pdf_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    print("\n--- Phase 2: Generating Centerline Reference GeoJSON ---")
    print("  > Step 2.1: Extracting alignment control points as GeoJSON...")
    prompt = (
        "From the 'Survey Control/Monument Perpetuation' sheets or similar alignment data tables, extract the points that define the mainline alignment geometry. "
        "These are the points that have both stationing and coordinate values. "
        "The output must be a GeoJSON FeatureCollection. Each feature's properties must include its station, and its geometry coordinates must be the ground coordinates (Easting, Northing)."
    )
    ground_geojson = call_gemini_api(pdf_path, prompt, is_json=True)
    print("  > Step 2.2: Performing geodetic conversion...")
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
        if not geometry or geometry.get('type') != 'Point': continue
        coords_ground = geometry.get('coordinates')
        if not isinstance(coords_ground, list) or len(coords_ground) < 2: continue
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
    print(f"  > Centerline generated with {processed_count} processed features.")
    return ground_geojson

# --- Phase 3: Feature Point Generation ---

def parse_station_value_from_props(prop: dict) -> Optional[float]:
    station_prop = prop.get('station') or prop.get('sta') or prop.get('STATION')
    if station_prop:
        parsed_station = parse_station_string(station_prop)
        if parsed_station: return parsed_station
    name = str(prop.get('name', ''))
    description = str(prop.get('description', ''))
    station_match = re.search(r'STA\.?\s*(\d{1,4})\+(\d{2}(\.\d+)?)', name + description, re.IGNORECASE)
    if station_match:
        return float(f"{station_match.group(1)}{station_match.group(2)}")
    try: return float(name)
    except (ValueError, TypeError): return None

def parse_station_string(station_val) -> Optional[float]:
    if station_val is None: return None
    if isinstance(station_val, (int, float)): return float(station_val)
    if isinstance(station_val, str):
        try: return float(station_val.replace('+', '')) if '+' in station_val else float(station_val)
        except (ValueError, TypeError): return None
    return None

def parse_offset_value(item: dict) -> Tuple[Optional[float], bool]:
    offset_str = str(item.get('offset', '')).lower().strip()
    dist_cl = item.get('distance_centerline')
    direction = -1 if 'lt' in offset_str else 1
    distance_val = None
    try: distance_val = float(dist_cl)
    except (ValueError, TypeError):
        if offset_str:
            match = re.search(r'(\d+(\.\d+)?)', offset_str)
            if match: distance_val = float(match.group(1))
            elif offset_str == '0': distance_val = 0.0
    if distance_val is not None:
        return (distance_val * direction, True) if distance_val != 0.0 else (0.0, True)
    return (None, False)

def prepare_centerline_data(centerline_geojson: dict) -> list:
    centerline_points = []
    for feature in centerline_geojson.get('features', []):
        props = feature.get('properties', {})
        station = parse_station_value_from_props(props)
        easting = props.get('calculated_grid_easting')
        northing = props.get('calculated_grid_northing')
        if station is not None and easting is not None and northing is not None:
            centerline_points.append((station, (easting, northing)))
    if not centerline_points:
        raise WorkflowError("No valid centerline points with stationing could be extracted from the reference GeoJSON.")
    centerline_points.sort(key=lambda p: p[0])
    print(f"  > Successfully prepared and sorted {len(centerline_points)} centerline points by station.")
    return centerline_points

def find_point_from_station_offset(station: float, grid_offset: float, sorted_centerline: list) -> tuple:
    if not sorted_centerline: raise WorkflowError("Centerline is empty, cannot calculate offset.")
    if station < sorted_centerline[0][0] or station > sorted_centerline[-1][0]:
        print(f"    - Warning: Station {station} is outside the centerline range ({sorted_centerline[0][0]} - {sorted_centerline[-1][0]}). Extrapolating.")
    p1_station, p1_coords = sorted_centerline[0]
    p2_station, p2_coords = sorted_centerline[-1]
    for i in range(len(sorted_centerline) - 1):
        if sorted_centerline[i][0] <= station <= sorted_centerline[i+1][0]:
            p1_station, p1_coords = sorted_centerline[i]
            p2_station, p2_coords = sorted_centerline[i+1]
            break
    segment_vec_x = p2_coords[0] - p1_coords[0]
    segment_vec_y = p2_coords[1] - p1_coords[1]
    station_diff = p2_station - p1_station
    interp_ratio = (station - p1_station) / station_diff if station_diff != 0 else 0
    cl_easting = p1_coords[0] + interp_ratio * segment_vec_x
    cl_northing = p1_coords[1] + interp_ratio * segment_vec_y
    segment_length = math.hypot(segment_vec_x, segment_vec_y)
    if segment_length == 0: return (cl_easting, cl_northing)
    perp_vec_x = segment_vec_y / segment_length
    perp_vec_y = -segment_vec_x / segment_length
    return (cl_easting + grid_offset * perp_vec_x, cl_northing + grid_offset * perp_vec_y)

def find_value_from_keys(data_dict: dict, keys: list):
    for key in keys:
        if key in data_dict: return data_dict[key]
    return None

def phase3_generate_feature_points(pdf_path: str, centerline_geojson: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    print("\n--- Phase 3: Generating Feature Points from Station and Offset ---")
    print("  > Step 3.1: Extracting feature attributes...")
    prompt = (
        "From the plan sheets, signs summary, and other detail sheets, extract features and items (like signs, culverts, guardrail) that are referenced to the mainline alignment stationing. "
        "Ignore features on side roads if they use a different stationing system. "
        "The output must be a JSON array where each object has properties for 'name', 'description', 'station', 'notes' and a clear 'offset' value like '15\\' Lt.' or '20\\' Rt.'. when the 'offset' value is unclear, assign a value that is just off the edge of the road based on the typical section for that area and note that in the array in the property 'notes'"
    )
    station_offset_data = call_gemini_api(pdf_path, prompt, is_json=True)
    if not isinstance(station_offset_data, list):
        raise WorkflowError("Feature attribute data is not a valid JSON array.")
    print("  > Step 3.2: Performing COGO calculations...")
    sorted_centerline = prepare_centerline_data(centerline_geojson)
    source_crs_epsg = int(metadata['source_crs_epsg'])
    combined_factor = metadata['combined_factor']
    target_crs_epsg = 4326
    transformer = Transformer.from_crs(CRS.from_epsg(source_crs_epsg), CRS.from_epsg(target_crs_epsg), always_xy=True)
    output_features = []
    STATION_KEYS = ["station", "centerline_distance", "sta"]
    for i, item in enumerate(station_offset_data):
        raw_station = find_value_from_keys(item, STATION_KEYS)
        station = parse_station_string(raw_station)
        ground_offset, offset_success = parse_offset_value(item)
        if station is None or not offset_success:
            desc = item.get('name', f'Item {i+1}')
            reason = "missing station" if station is None else "ambiguous offset"
            print(f"    - Skipping '{desc}': {reason}.")
            continue
        try:
            grid_offset = ground_offset / combined_factor
            grid_easting, grid_northing = find_point_from_station_offset(station, grid_offset, sorted_centerline)
            lon, lat = transformer.transform(grid_easting, grid_northing)
            output_features.append({
                "type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {**item, "parsed_station": station, "parsed_ground_offset_ft": ground_offset, "calculated_grid_offset_ft": grid_offset, "calculated_grid_easting": grid_easting, "calculated_grid_northing": grid_northing}
            })
        except Exception as e:
            print(f"    - Error processing item {i+1} (Station: {station}): {e}")
    final_geojson = {"type": "FeatureCollection", "name": "Generated Points from Station-Offset", "crs": {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{target_crs_epsg}"}}, "features": output_features}
    print(f"  > COGO calculations complete. Generated {len(output_features)} features.")
    return final_geojson

# --- NEW FUNCTION FOR KMZ CREATION ---
def create_kmz_file(geojson_data: Dict[str, Any], output_filename: str):
    """Creates a KMZ file from a GeoJSON FeatureCollection of points."""
    kml = simplekml.Kml(name=geojson_data.get("name", "Generated Features"))

    for i, feature in enumerate(geojson_data.get('features', [])):
        props = feature.get('properties', {})
        geom = feature.get('geometry', {})
        coords = geom.get('coordinates')

        if geom.get('type') != 'Point' or not coords or len(coords) < 2:
            continue

        pnt = kml.newpoint(name=props.get('name', f'Feature {i+1}'))
        
        # Create a nice HTML description for the balloon
        desc = "<table>"
        for key, value in props.items():
            desc += f"<tr><td><b>{key}:</b></td><td>{value}</td></tr>"
        desc += "</table>"
        pnt.description = desc
        
        pnt.coords = [(coords[0], coords[1])]  # (lon, lat)

    print(f"  > Creating KMZ file with {len(kml.features)} placemarks...")
    kml.savekmz(output_filename)

# --- Phase 4: Validation and Output (MODIFIED) ---

def phase4_save_outputs(centerline_geojson: Dict, final_points_geojson: Dict):
    """Saves the final GeoJSON and KMZ files and provides validation instructions."""
    print("\n--- Phase 4: Saving Outputs and Validation ---")
    
    centerline_filename = 'output_centerline_wgs84.geojson'
    final_points_filename = 'output_generated_points.geojson'
    final_kmz_filename = 'output_generated_points.kmz'
    
    try:
        # Save the GeoJSON files
        with open(centerline_filename, 'w') as f: json.dump(centerline_geojson, f, indent=2)
        print(f"  > Successfully saved reference centerline to: '{centerline_filename}'")
        
        with open(final_points_filename, 'w') as f: json.dump(final_points_geojson, f, indent=2)
        print(f"  > Successfully saved final feature points to: '{final_points_filename}'")
        
        # Create and save the KMZ file
        create_kmz_file(final_points_geojson, final_kmz_filename)
        print(f"  > Successfully saved KMZ file to: '{final_kmz_filename}'")
        
        print("\n--- Validation ---")
        print("Process complete. You can validate the results in two ways:")
        print(f"  1. Drag and drop '{final_points_filename}' into a tool like https://geojson.io/")
        print(f"  2. Open '{final_kmz_filename}' in Google Earth Pro.")

    except Exception as e:
        raise WorkflowError(f"Failed to write output files: {e}")

# --- Main Execution ---

def main():
    print("=============================================")
    print(" Automated PDF Plan Set Processing Workflow ")
    print("=============================================")
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
        project_metadata = phase1_extract_metadata(pdf_path)
        
        print("\n--- User Validation Required ---")
        print(f"  > AI extracted the following metadata:")
        print(f"    - EPSG Code: {project_metadata.get('source_crs_epsg')}")
        print(f"    - Combined Factor: {project_metadata.get('combined_factor')}")
        confirm = input("  > Does this information look correct? (yes/no): ").lower().strip()
        if confirm not in ['y', 'yes']:
            print("  > Aborting workflow based on user input.")
            sys.exit(1)
            
        centerline_geojson = phase2_generate_centerline(pdf_path, project_metadata)
        final_points_geojson = phase3_generate_feature_points(pdf_path, centerline_geojson, project_metadata)
        phase4_save_outputs(centerline_geojson, final_points_geojson)
    except WorkflowError as e:
        print(f"\nA workflow error occurred: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()