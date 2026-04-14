# PDF Plan Set Geolocation Tool
# AKA the 🤖 Construct-o-matic AI Tool 🤖

A Python tool that builds a **geospatial feature inventory** from construction plan set PDFs. It uses the Google Gemini API to extract features from 9 sheet categories — roadway summary, pipe/culvert summary, plan & profile, drainage, signing, pavement marking, erosion control, traffic control, and utility plans — then converts them into geo-referenced Points and LineStrings viewable in Google Earth or any GIS platform.

This tool was developed to solve the challenge of unlocking valuable data trapped in static PDF documents, providing a scalable method to link construction details directly to systems like Pavement Management (PMS) and asset inventories.

## Core Workflow

The script (`process_plans_v7.py`) operates in 7 phases, leveraging the Google Gemini API for data extraction and Python libraries for geodetic calculations and COGO. Phases 2 & 4 run in parallel, and all Phase 5 extractions run concurrently via `ThreadPoolExecutor`.

#### Phase 1: Geodetic Parameter Extraction
*   Sends the PDF to the Gemini API to extract the source CRS (e.g., `EPSG:2241`), the project's Combined Factor, station equations, and project metadata (name, route, station range).
*   Presents extracted parameters to the user for validation before proceeding.

#### Phase 2: Typical Section Extraction
*   Extracts cross-section dimensions from the Typical Section sheets — lane widths, shoulder widths, curb-to-curb width, sidewalk width, and right-of-way offsets.
*   Used to assign default offsets when a feature's offset isn't explicitly specified in the plans. If extraction fails, falls back to hard-coded defaults (24ft curb-to-curb, 4ft shoulders).

#### Phase 3: Centerline Generation
*   Extracts alignment control points (with stationing and ground coordinates) from survey control sheets.
*   Converts ground coordinates to grid coordinates (via Combined Factor) and then to WGS84 lat/lon using `pyproj`.
*   Generates a **LineString** connecting all control points in station order — the project centerline rendered as an actual line, not just scattered pins.

#### Phase 4: Sheet Inventory & Classification *(runs in parallel with Phase 2)*
*   Classifies every sheet in the plan set by title.
*   Uses **fuzzy keyword matching** to map sheet titles to extraction configs (e.g., "Plan/ Profile Sheets", "Plan Sheet", and "Plan & Profile" all match the same config).
*   Determines which of the 9 extraction prompts to run in Phase 5.

#### Phase 5: Category-Specific Feature Extraction *(parallel)*
*   Launches **all matched extraction prompts concurrently** via `ThreadPoolExecutor`:
    *   **Roadway Summary:** every pay item with station ranges (excavation, paving, fencing, etc.)
    *   **Pipe & Culvert Summary:** all pipes/culverts with size, material, and station
    *   **Plan & Profile:** curb & gutter, sidewalk, guardrail, approaches
    *   **Drainage:** inlets, manholes, pipe runs, swales, ponds, culverts
    *   **Signing:** all signs with MUTCD codes
    *   **Pavement Markings:** striping lines, symbols, crosswalks
    *   **Erosion Control:** silt fence, inlet protection, wattles, erosion blanket
    *   **Traffic Control:** temporary signs, barricades, flagging stations
    *   **Utility Plans:** utility relocations, crossings, conduit runs
*   Each prompt requests structured data with `geometry_type` (point/line), `start_station`, `end_station`, `offset`, and `category`.
*   Only runs prompts for sheet types that exist in the plan set (typically 6-8 of 9).

#### Phase 6: Geometry Construction
*   Routes each feature to the appropriate geometry builder:
    *   **Points:** single station + offset (signs, inlets, manholes, approaches)
    *   **LineStrings:** start/end station + offset, interpolated every 25ft along the centerline (curb & gutter, guardrail, striping, silt fence, pipe runs)
*   All geometry passes through the same COGO pipeline: station → linear interpolation along centerline → perpendicular offset → grid-to-WGS84 transform.
*   **Note:** The centerline is built from sparse control points (typically 6-10 over the project length) with linear interpolation between them. On curved alignments, LineStrings will approximate the arc as straight segments between control points rather than following the true curve geometry.

#### Phase 7: Styled Output
*   Saves GeoJSON and a **styled KMZ** with:
    *   Color-coded folders per category (drainage in blue, guardrail in orange-red, signing in gold, etc.)
    *   LineStrings rendered as colored lines along the road
    *   Polygons with semi-transparent fill
    *   Centerline as a red line
*   All outputs are organized in a unique folder: `outputs/<planset_name>_<timestamp>/`

## Category Color Coding

| Category | Color | Typical Geometry |
|---|---|---|
| Centerline | Red | LineString |
| Drainage | Blue | Point + LineString |
| Signing | Gold | Point |
| Pavement Marking | White | LineString |
| Curb & Gutter | Gray | LineString |
| Sidewalk | Silver | LineString |
| Guardrail | Orange-Red | LineString |
| Approach | Brown | Point |
| Erosion Control | Green | LineString + Point |
| Traffic Control | Orange | Point |
| Survey Monument | Magenta | Point |
| Utility | Purple | Point + LineString |
| Roadway Surface | Dark Gray | LineString |
| Other | Gray | Point |

## Prerequisites
*   Python 3.8 or newer.
*   A valid Google AI API Key with access to the Gemini family of models.
*   The script defaults to the `gemini-3.1-pro-preview` model. To use a different model, edit the `MODEL_NAME` variable at the top of the script.

## Installation & Setup

**1. Clone the Repository:**
```bash
git clone https://github.com/Mike-Copeland/construct-o-matic.git
cd construct-o-matic
```

**2. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**3. Configure API Key:**
Set your Google AI API Key as an environment variable named `GOOGLE_API_KEY`.

If the environment variable is not set, the script will prompt you to enter the key manually upon execution.

## Usage

```bash
python process_plans_v7.py "path/to/your/project_plans.pdf"
```

The script prints progress for each phase. You will be prompted to validate the extracted geodetic parameters before processing continues.

## Expected Output

Upon successful execution, a timestamped output folder is created:

```
outputs/
  23204_Final_Plans_20260414_153022/
    23204_Final_Plans_20260414_153022_centerline.geojson
    23204_Final_Plans_20260414_153022_digital_twin.geojson
    23204_Final_Plans_20260414_153022_digital_twin.kmz
```

*   **`_centerline.geojson`** — Geo-referenced project centerline (control points + LineString).
*   **`_digital_twin.geojson`** — All extracted features with Point, LineString, and Polygon geometries.
*   **`_digital_twin.kmz`** — Styled KMZ with category folders for Google Earth Pro.

Validate by opening the KMZ in Google Earth Pro or dragging the GeoJSON into [geojson.io](https://geojson.io/).

## Test Results

Tested against 3 ITD plan sets:

| Plan Set | Sheets | Features Extracted | Points | Lines | Skipped |
|---|---|---|---|---|---|
| 22217 (SH-77, 160 sta) | 25 + 18 survey | 304 raw → 242 output | 165 | 77 | 62 |

Features by category (22217):
- Drainage: 75 points + 37 lines (inlets, manholes, pipe runs)
- Roadway Surface: 16 lines (paving, aggregate base with station ranges)
- Pavement Markings: 3 points + 19 lines (striping, symbols)
- Signing: 18 points
- Other: 62 points (roadway summary items)
- Approaches: 7 points + 5 lines

*Skipped items are project-wide pay items from the Roadway Summary that have quantities but no specific station — expected behavior.*

## Limitations and Disclaimer

*   This is a proof-of-concept tool primarily tested with Idaho Transportation Department (ITD) plan sets. The prompts may require modification for plan sets from other agencies or with different formatting.
*   **Curve accuracy:** The centerline is linearly interpolated between sparse control points. Features on curved alignments will be placed along straight-line approximations, not the true arc. Accuracy improves with more control points in the source plans.
*   **Sheet coverage:** Extracts from 9 sheet categories: Roadway Summary, Pipe/Culvert Summary, Plan & Profile, Drainage, Signing, Pavement Marking, Erosion Control, Traffic Control, and Utility Plans. Bridge/Structure, Right of Way, Cross Section, and Landscaping sheets are inventoried but not yet extracted.
*   **No elevation data:** All output geometry is 2D (lat/lon). Profile/elevation information from Plan & Profile sheets is not captured.
*   **Roadway Summary items without stations:** Project-wide lump-sum pay items (mobilization, traffic control lump sum, etc.) are extracted but skipped during geometry construction since they have no station reference.
*   Accuracy depends on the quality of the source PDF and the correctness of the AI-extracted parameters. User validation in Phase 1 is critical.
*   Users are responsible for ensuring that their use of this tool complies with all policies of their respective organizations.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
