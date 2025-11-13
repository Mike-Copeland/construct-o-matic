# PDF Plan Set Geolocation Tool (🤖 Construct-o-matic 🤖)

This repository contains a Python-based workflow for extracting and geolocating project features from construction plan set PDFs. The primary function is to convert project-level station and offset data into geographic coordinates (latitude/longitude), making legacy and ongoing project data compatible with asset management and GIS platforms.

This tool was developed to solve the challenge of unlocking valuable data trapped in static PDF documents, providing a scalable method to link construction details directly to systems like Pavement Management (PMS) and asset inventories.

## Core Workflow

The script operates in a sequential, multi-phase process that leverages the Google Gemini API for data extraction and Python libraries for geodetic calculations.

#### Phase 1: Geodetic Parameter Extraction
*   **Action:** The script sends the initial pages of the PDF plan set to the Gemini API.
*   **Objective:** Extract foundational geodetic parameters required for all subsequent calculations, including the source Coordinate Reference System (e.g., `EPSG:2241`), the project's Combined Factor, and any specified station equations.
*   **User Interaction:** The script presents the extracted parameters to the user for validation before proceeding, ensuring the baseline for calculations is correct.

#### Phase 2: Centerline Geo-referencing
*   **Action:** The script prompts the Gemini API to identify and extract alignment control points (which have both stationing and ground coordinates) from the plan tables.
*   **Objective:** A Python function then applies the Combined Factor to convert these ground coordinates to grid coordinates. The `pyproj` library is used to transform the grid coordinates into the WGS84 geographic coordinate system (EPSG:4326), creating a geo-referenced digital centerline.

#### Phase 3: Feature Geolocation via COGO
*   **Action:** The Gemini API is prompted to parse the plan set for features (e.g., signs, culverts, guardrail) and their associated station and offset values.
*   **Objective:** A Python-based Coordinate Geometry (COGO) function processes this extracted text. It calculates the precise grid coordinates for each feature by interpolating its position along the geo-referenced centerline and applying the scaled grid offset. These grid coordinates are then transformed into latitude and longitude.

#### Phase 4: Data Export
*   **Action:** The script compiles the processed data into standard geospatial formats.
*   **Objective:** Save the geo-referenced centerline and the final feature points as separate GeoJSON files. Additionally, a KMZ file of the feature points is created for easy visualization in Google Earth Pro.

## Prerequisites
*   Python 3.8 or newer.
*   A valid Google AI API Key with access to the Gemini family of models.

## Installation & Setup

**1. Clone the Repository:**
Clone this repository to your local machine or download the files as a ZIP.

**2. Install Dependencies:**
Navigate to the project directory in your terminal and install the required Python libraries from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

**3. Configure API Key:**
For security and convenience, it is highly recommended to set your Google AI API Key as an environment variable named `GOOGLE_API_KEY`.

If the environment variable is not set, the script will prompt you to enter the key manually in the terminal upon first execution.

## Usage

Execute the script from your terminal. The sole required argument is the file path to the PDF plan set you wish to process.

```bash
python process_plans.py "C:\path\to\your\project_plans.pdf"
```

The script will print its progress for each phase to the console. Be prepared to provide input when prompted to validate the extracted geodetic parameters.

## Expected Output

Upon successful execution, the following files will be created in the same directory as the script:
*   `output_centerline_wgs84.geojson`: The geo-referenced project centerline.
*   `output_generated_points.geojson`: The geolocated feature points.
*   `output_generated_points.kmz`: The feature points in KMZ format for use in Google Earth.

You can validate the output by loading the GeoJSON files into GIS software (like QGIS or ArcGIS Pro) or by opening the KMZ file.

## Limitations and Disclaimer

*   This is a proof-of-concept tool and has been primarily tested with Idaho Transportation Department (ITD) plan sets. The prompts used to query the Gemini API may require modification to work effectively with plan sets from other agencies or with different formatting.
*   The accuracy of the output is directly dependent on the quality of the data in the source PDF and the correctness of the initial parameters extracted by the AI. User validation in Phase 1 is critical.
*   Users are responsible for ensuring that their use of this tool and the data it processes complies with all policies of their respective organizations.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
