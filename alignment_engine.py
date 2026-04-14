# alignment_engine.py — Precise horizontal + vertical alignment geometry engine
# Computes station+offset → grid coords (easting, northing, elevation)
# Uses horizontal curve PI data for true arc geometry and vertical VPI data for profile elevations.

import math
import re
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

# --- Data Structures ---

@dataclass
class HorizontalPI:
    station: float
    easting: float        # ground coordinates
    northing: float
    delta_rad: float      # deflection angle in radians (+ = RT/clockwise, - = LT/counter-clockwise)
    radius: float         # curve radius in feet
    tangent_length: float
    curve_length: float
    pc_station: float
    pt_station: float
    superelevation: Optional[float] = None  # as decimal (0.052 = 5.2%)

@dataclass
class AlignmentSegment:
    start_station: float
    end_station: float
    segment_type: str         # "tangent" or "curve"
    start_easting: float
    start_northing: float
    bearing: float            # azimuth in radians (from north, clockwise)
    # Curve-specific fields
    center_easting: float = 0.0
    center_northing: float = 0.0
    radius: float = 0.0
    start_angle: float = 0.0  # angle from center to PC, measured from north CW
    delta_rad: float = 0.0
    direction: int = 0        # +1 = RT (CW), -1 = LT (CCW)

@dataclass
class VerticalPI:
    station: float
    elevation: float
    curve_length: float = 0.0  # 0 = no vertical curve at this VPI
    k_value: float = 0.0
    vpc_station: float = 0.0
    vpt_station: float = 0.0
    vpc_elevation: Optional[float] = None
    vpt_elevation: Optional[float] = None

@dataclass
class TypicalSection:
    label: str
    start_station: float
    end_station: float
    lane_width_left: float = 12.0
    lane_width_right: float = 12.0
    shoulder_left: float = 4.0
    shoulder_right: float = 4.0
    curb_to_curb: float = 24.0
    sidewalk_width: float = 0.0
    row_offset_left: float = 50.0
    row_offset_right: float = 50.0

# --- Parsing Helpers ---

def parse_dms_angle(dms_str: str) -> Optional[float]:
    """Parse a DMS angle string like '21°27'17\" LT' to radians. + = RT, - = LT."""
    if not dms_str:
        return None
    # Handle various Unicode degree symbols and formats
    cleaned = dms_str.replace('\u00b0', '°').replace('\u2019', "'").replace('\u201d', '"')
    match = re.search(
        r"(\d+)\s*[°]\s*(\d+)\s*['′]\s*(\d+(?:\.\d+)?)\s*[\"″]?\s*(LT|RT|L|R|LEFT|RIGHT)",
        cleaned, re.IGNORECASE
    )
    if not match:
        # Try decimal degrees format
        dec_match = re.search(r"([\d.]+)\s*[°]?\s*(LT|RT|L|R)", cleaned, re.IGNORECASE)
        if dec_match:
            degrees = float(dec_match.group(1))
            direction = dec_match.group(2).upper()
            rad = math.radians(degrees)
            return rad if direction.startswith('R') else -rad
        return None

    d, m, s = int(match.group(1)), int(match.group(2)), float(match.group(3))
    direction = match.group(4).upper()
    degrees = d + m / 60.0 + s / 3600.0
    rad = math.radians(degrees)
    return rad if direction.startswith('R') else -rad

def parse_numeric(val) -> Optional[float]:
    """Parse a numeric value, stripping unit suffixes like feet marks."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        cleaned = val.replace("'", "").replace('"', '').replace(',', '').strip()
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return None
    return None

def parse_station(val) -> Optional[float]:
    """Parse station string like '128+39.42' to float 12839.42."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val.replace('+', '')) if '+' in val else float(val)
        except (ValueError, TypeError):
            return None
    return None

# --- Alignment Engine ---

class AlignmentEngine:
    """Precise station+offset → coordinate conversion using horizontal curves and vertical profile."""

    def __init__(self):
        self.horizontal_pis: List[HorizontalPI] = []
        self.segments: List[AlignmentSegment] = []
        self.vertical_pis: List[VerticalPI] = []
        self.typical_sections: List[TypicalSection] = []
        self.begin_station: float = 0.0
        self.end_station: float = 0.0
        self.combined_factor: float = 1.0
        self._grades: List[Tuple[float, float, float]] = []  # (start_sta, end_sta, grade)

    # --- Build from PI Data ---

    def build_horizontal(self, pi_data: List[dict], begin_station: float, end_station: float):
        """Build horizontal alignment segments from a list of PI dictionaries."""
        self.begin_station = begin_station
        self.end_station = end_station

        # Parse PI data
        pis = []
        for pid in pi_data:
            delta = parse_dms_angle(str(pid.get('delta', '')))
            if delta is None:
                continue
            pi = HorizontalPI(
                station=parse_station(pid.get('pi_station', pid.get('PI_station'))) or 0,
                easting=parse_numeric(pid.get('easting', pid.get('E_coord'))) or 0,
                northing=parse_numeric(pid.get('northing', pid.get('N_coord'))) or 0,
                delta_rad=delta,
                radius=parse_numeric(pid.get('radius')) or 0,
                tangent_length=parse_numeric(pid.get('tangent_length', pid.get('tangent'))) or 0,
                curve_length=parse_numeric(pid.get('curve_length', pid.get('length'))) or 0,
                pc_station=parse_station(pid.get('pc_station', pid.get('PC_station'))) or 0,
                pt_station=parse_station(pid.get('pt_station', pid.get('PT_station'))) or 0,
            )
            # Compute PC/PT if not provided
            if pi.pc_station == 0 and pi.tangent_length > 0:
                pi.pc_station = pi.station - pi.tangent_length
            if pi.pt_station == 0 and pi.curve_length > 0:
                pi.pt_station = pi.pc_station + pi.curve_length

            se = pid.get('superelevation', '')
            if se and isinstance(se, str):
                match = re.search(r'([\d.]+)', se)
                if match:
                    pi.superelevation = float(match.group(1)) / 100.0

            pis.append(pi)

        pis.sort(key=lambda p: p.station)
        self.horizontal_pis = pis

        if len(pis) < 1:
            print("  > WARNING: No horizontal PIs found. Alignment engine has no curve data.")
            return

        # Build segments: tangent-curve-tangent-curve-...
        self.segments = []
        self._build_segments()
        print(f"  > Built {len(self.segments)} alignment segments ({sum(1 for s in self.segments if s.segment_type == 'curve')} curves, "
              f"{sum(1 for s in self.segments if s.segment_type == 'tangent')} tangents)")

    def _build_segments(self):
        """Build ordered tangent and curve segments from PI data."""
        pis = self.horizontal_pis
        if not pis:
            return

        # Compute tangent bearings between consecutive PIs
        bearings = []
        for i in range(len(pis) - 1):
            dE = pis[i + 1].easting - pis[i].easting
            dN = pis[i + 1].northing - pis[i].northing
            bearing = math.atan2(dE, dN)  # azimuth from north, CW positive
            bearings.append(bearing)

        # For the last PI, the outgoing bearing is the incoming bearing deflected by delta
        if bearings:
            bearings.append(bearings[-1] + pis[-1].delta_rad)
        else:
            bearings.append(0.0)

        # Incoming bearing for the first PI
        # The first tangent starts before the first PI
        # The incoming bearing to the first PI is: bearing[0] (bearing from PI[0] to PI[1])
        # but the actual incoming tangent has been deflected by PI[0]'s delta
        incoming_bearing = bearings[0] - pis[0].delta_rad if pis[0].delta_rad != 0 else bearings[0]

        # Build the segment list
        current_station = self.begin_station
        current_bearing = incoming_bearing

        for i, pi in enumerate(pis):
            # Tangent before this curve (from current_station to PC)
            if pi.pc_station > current_station + 0.1:
                # Compute start point of this tangent
                if self.segments:
                    # Start from end of previous segment
                    prev = self.segments[-1]
                    start_e, start_n = self._segment_end_point(prev)
                else:
                    # Project backwards from PI along incoming tangent
                    dist_to_pi = pi.station - current_station
                    start_e = pi.easting - dist_to_pi * math.sin(current_bearing)
                    start_n = pi.northing - dist_to_pi * math.cos(current_bearing)

                self.segments.append(AlignmentSegment(
                    start_station=current_station,
                    end_station=pi.pc_station,
                    segment_type="tangent",
                    start_easting=start_e,
                    start_northing=start_n,
                    bearing=current_bearing,
                ))

            # Curve at this PI
            direction = 1 if pi.delta_rad > 0 else -1  # RT = CW = +1, LT = CCW = -1
            abs_delta = abs(pi.delta_rad)

            # PC point: project back from PI along incoming tangent
            pc_e = pi.easting - pi.tangent_length * math.sin(current_bearing)
            pc_n = pi.northing - pi.tangent_length * math.cos(current_bearing)

            # Curve center: perpendicular from PC, toward center of curvature
            # For RT curve: center is to the right of travel direction
            # For LT curve: center is to the left
            perp_bearing = current_bearing + direction * math.pi / 2
            center_e = pc_e + pi.radius * math.sin(perp_bearing)
            center_n = pc_n + pi.radius * math.cos(perp_bearing)

            # Start angle: angle from center to PC
            start_angle = math.atan2(pc_e - center_e, pc_n - center_n)

            self.segments.append(AlignmentSegment(
                start_station=pi.pc_station,
                end_station=pi.pt_station,
                segment_type="curve",
                start_easting=pc_e,
                start_northing=pc_n,
                bearing=current_bearing,
                center_easting=center_e,
                center_northing=center_n,
                radius=pi.radius,
                start_angle=start_angle,
                delta_rad=pi.delta_rad,
                direction=direction,
            ))

            # Update bearing: outgoing tangent bearing = incoming + delta
            current_bearing = current_bearing + pi.delta_rad
            current_station = pi.pt_station

        # Final tangent after the last curve
        if current_station < self.end_station - 0.1:
            if self.segments:
                prev = self.segments[-1]
                start_e, start_n = self._segment_end_point(prev)
            else:
                start_e, start_n = 0.0, 0.0

            self.segments.append(AlignmentSegment(
                start_station=current_station,
                end_station=self.end_station,
                segment_type="tangent",
                start_easting=start_e,
                start_northing=start_n,
                bearing=current_bearing,
            ))

    def _segment_end_point(self, seg: AlignmentSegment) -> Tuple[float, float]:
        """Compute the endpoint (easting, northing) of a segment."""
        if seg.segment_type == "tangent":
            dist = seg.end_station - seg.start_station
            return (
                seg.start_easting + dist * math.sin(seg.bearing),
                seg.start_northing + dist * math.cos(seg.bearing),
            )
        else:  # curve
            end_angle = seg.start_angle + seg.delta_rad
            return (
                seg.center_easting + seg.radius * math.sin(end_angle),
                seg.center_northing + seg.radius * math.cos(end_angle),
            )

    def _find_segment(self, station: float) -> Optional[AlignmentSegment]:
        """Find the segment containing the given station."""
        for seg in self.segments:
            if seg.start_station - 0.1 <= station <= seg.end_station + 0.1:
                return seg
        # Extrapolate from nearest segment
        if self.segments:
            if station < self.segments[0].start_station:
                return self.segments[0]
            if station > self.segments[-1].end_station:
                return self.segments[-1]
        return None

    # --- Core Computation: Station → Grid Coordinates ---

    def station_to_grid_coords(self, station: float) -> Tuple[float, float]:
        """Compute ground easting/northing at a station on the centerline."""
        seg = self._find_segment(station)
        if seg is None:
            raise ValueError(f"No alignment segment found for station {station}")

        if seg.segment_type == "tangent":
            dist = station - seg.start_station
            e = seg.start_easting + dist * math.sin(seg.bearing)
            n = seg.start_northing + dist * math.cos(seg.bearing)
            return (e, n)
        else:  # curve
            arc_dist = station - seg.start_station
            angle = arc_dist / seg.radius * seg.direction
            e = seg.center_easting + seg.radius * math.sin(seg.start_angle + angle)
            n = seg.center_northing + seg.radius * math.cos(seg.start_angle + angle)
            return (e, n)

    def bearing_at_station(self, station: float) -> float:
        """Get the forward azimuth (bearing) at a station."""
        seg = self._find_segment(station)
        if seg is None:
            return 0.0
        if seg.segment_type == "tangent":
            return seg.bearing
        else:  # curve — tangent to the arc at this point
            arc_dist = station - seg.start_station
            angle = arc_dist / seg.radius * seg.direction
            # Tangent direction is perpendicular to radial direction
            radial_angle = seg.start_angle + angle
            # For RT curve: tangent = radial + 90° CW
            # For LT curve: tangent = radial - 90° CCW
            return radial_angle + seg.direction * math.pi / 2

    def station_offset_to_grid_coords(self, station: float, offset: float) -> Tuple[float, float]:
        """Compute ground easting/northing at a station with perpendicular offset.
        Offset: positive = right of CL, negative = left of CL (looking ahead station)."""
        cl_e, cl_n = self.station_to_grid_coords(station)
        bearing = self.bearing_at_station(station)

        # Perpendicular to the right = bearing + 90°
        perp_bearing = bearing + math.pi / 2
        e = cl_e + offset * math.sin(perp_bearing)
        n = cl_n + offset * math.cos(perp_bearing)
        return (e, n)

    # --- Vertical Profile ---

    def build_vertical(self, vpi_data: List[dict]):
        """Build vertical profile from VPI data."""
        vpis = []
        for vd in vpi_data:
            sta = parse_station(vd.get('vpi_station', vd.get('VPI_station')))
            elev = parse_numeric(vd.get('vpi_elevation', vd.get('VPI_elevation')))
            if sta is None or elev is None:
                continue
            cl = parse_numeric(vd.get('curve_length', vd.get('length'))) or 0
            vpi = VerticalPI(
                station=sta,
                elevation=elev,
                curve_length=cl,
                k_value=parse_numeric(vd.get('k_value', vd.get('K_value'))) or 0,
                vpc_station=parse_station(vd.get('vpc_station', vd.get('VPC_station'))) or (sta - cl / 2 if cl > 0 else sta),
                vpt_station=parse_station(vd.get('vpt_station', vd.get('VPT_station'))) or (sta + cl / 2 if cl > 0 else sta),
                vpc_elevation=parse_numeric(vd.get('vpc_elevation', vd.get('VPC_elevation'))),
                vpt_elevation=parse_numeric(vd.get('vpt_elevation', vd.get('VPT_elevation'))),
            )
            vpis.append(vpi)

        vpis.sort(key=lambda v: v.station)
        self.vertical_pis = vpis

        if len(vpis) < 2:
            print(f"  > Vertical profile: {len(vpis)} VPIs (insufficient for profile computation)")
            return

        # Compute grades between consecutive VPIs
        self._grades = []
        for i in range(len(vpis) - 1):
            dSta = vpis[i + 1].station - vpis[i].station
            if dSta > 0:
                grade = (vpis[i + 1].elevation - vpis[i].elevation) / dSta
                self._grades.append((vpis[i].station, vpis[i + 1].station, grade))

        # Compute VPC elevations where missing
        for i, vpi in enumerate(vpis):
            if vpi.curve_length > 0 and vpi.vpc_elevation is None:
                # Find incoming grade
                g_in = self._grade_at(vpi.station - 1)
                if g_in is not None:
                    vpi.vpc_elevation = vpi.elevation - g_in * (vpi.curve_length / 2)

        print(f"  > Vertical profile: {len(vpis)} VPIs, {len(self._grades)} grade segments")

    def _grade_at(self, station: float) -> Optional[float]:
        """Get the tangent grade at a station (between VPIs)."""
        for start, end, grade in self._grades:
            if start <= station <= end:
                return grade
        return None

    def station_to_elevation(self, station: float) -> Optional[float]:
        """Compute design elevation at a station using vertical curve parabolic interpolation."""
        if not self.vertical_pis:
            return None

        # Check if station is within a vertical curve
        for i, vpi in enumerate(self.vertical_pis):
            if vpi.curve_length > 0:
                vpc_sta = vpi.vpc_station
                vpt_sta = vpi.vpt_station
                if vpc_sta <= station <= vpt_sta:
                    # Station is on this vertical curve
                    # Get incoming and outgoing grades
                    g1 = self._grades[i - 1][2] if i > 0 else 0
                    g2 = self._grades[i][2] if i < len(self._grades) else 0
                    L = vpi.curve_length

                    # VPC elevation
                    vpc_elev = vpi.vpc_elevation
                    if vpc_elev is None:
                        vpc_elev = vpi.elevation - g1 * (L / 2)

                    x = station - vpc_sta
                    # Parabolic formula: y = vpc_elev + g1*x + (g2-g1)/(2L) * x^2
                    elev = vpc_elev + g1 * x + (g2 - g1) / (2 * L) * x * x
                    return elev

        # Station is on a tangent grade — interpolate linearly
        for start, end, grade in self._grades:
            if start <= station <= end:
                # Find the VPI at the start
                for vpi in self.vertical_pis:
                    if abs(vpi.station - start) < 0.1:
                        return vpi.elevation + grade * (station - vpi.station)
        return None

    # --- Cross Section ---

    def build_cross_sections(self, section_data: List[dict]):
        """Build typical section model from section data."""
        sections = []
        for sd in section_data:
            start = parse_station(sd.get('applies_from_station', sd.get('station_range', [''])[0] if isinstance(sd.get('station_range'), list) else ''))
            end = parse_station(sd.get('applies_to_station'))
            # Try to parse station_range string like "Sta. 124+06 - 158+50"
            sr = sd.get('station_range', '')
            if isinstance(sr, str) and '-' in sr:
                parts = re.findall(r'(\d+\+[\d.]+)', sr)
                if len(parts) >= 2:
                    start = start or parse_station(parts[0])
                    end = end or parse_station(parts[1])

            sections.append(TypicalSection(
                label=sd.get('label', sd.get('section_name', 'Default')),
                start_station=start or self.begin_station,
                end_station=end or self.end_station,
                lane_width_left=parse_numeric(sd.get('lane_widths', {}).get('left_lane', sd.get('lane_widths', {}).get('left', 12))) or 12,
                lane_width_right=parse_numeric(sd.get('lane_widths', {}).get('right_lane', sd.get('lane_widths', {}).get('right', 12))) or 12,
                shoulder_left=parse_numeric(sd.get('shoulder_widths', {}).get('left_shoulder', sd.get('shoulder_widths', {}).get('left', 4))) or 4,
                shoulder_right=parse_numeric(sd.get('shoulder_widths', {}).get('right_shoulder', sd.get('shoulder_widths', {}).get('right', 4))) or 4,
                curb_to_curb=parse_numeric(sd.get('curb_to_curb_width', sd.get('overall_pavement_width', 24))) or 24,
                sidewalk_width=parse_numeric(sd.get('sidewalk_width', 0)) or 0,
                row_offset_left=parse_numeric(sd.get('right_of_way_width', {}).get('left', sd.get('right_of_way_offset_left_ft', 50))) or 50,
                row_offset_right=parse_numeric(sd.get('right_of_way_width', {}).get('right', sd.get('right_of_way_offset_right_ft', 50))) or 50,
            ))
        sections.sort(key=lambda s: s.start_station)
        self.typical_sections = sections
        print(f"  > Cross-section model: {len(sections)} typical section(s)")

    def get_section(self, station: float) -> TypicalSection:
        """Return the typical section applicable at a station."""
        for s in self.typical_sections:
            if s.start_station <= station <= s.end_station:
                return s
        if self.typical_sections:
            return self.typical_sections[0]
        return TypicalSection(label="Default", start_station=0, end_station=999999)

    # --- High-Level API ---

    def station_offset_to_wgs84(self, station: float, offset: float,
                                 combined_factor: float, transformer) -> Tuple[float, float, Optional[float]]:
        """Convert station+offset to WGS84 (lon, lat, elevation).
        offset: positive = right, negative = left."""
        # Ground coords → grid coords
        ground_e, ground_n = self.station_offset_to_grid_coords(station, offset / combined_factor * combined_factor)
        grid_e = ground_e / combined_factor
        grid_n = ground_n / combined_factor
        # Grid → WGS84
        lon, lat = transformer.transform(grid_e, grid_n)
        elev = self.station_to_elevation(station)
        return (lon, lat, elev)

    def generate_centerline_coords(self, combined_factor: float, transformer,
                                    spacing_ft: float = 25.0) -> List[List[float]]:
        """Generate dense centerline coordinates for KMZ visualization."""
        coords = []
        station = self.begin_station
        while station <= self.end_station:
            try:
                ground_e, ground_n = self.station_to_grid_coords(station)
                grid_e = ground_e / combined_factor
                grid_n = ground_n / combined_factor
                lon, lat = transformer.transform(grid_e, grid_n)
                elev = self.station_to_elevation(station)
                if elev is not None:
                    coords.append([lon, lat, elev])
                else:
                    coords.append([lon, lat])
            except Exception:
                pass
            station += spacing_ft
        # Always include exact end station
        if coords:
            try:
                ground_e, ground_n = self.station_to_grid_coords(self.end_station)
                grid_e = ground_e / combined_factor
                grid_n = ground_n / combined_factor
                lon, lat = transformer.transform(grid_e, grid_n)
                elev = self.station_to_elevation(self.end_station)
                if elev is not None:
                    coords.append([lon, lat, elev])
                else:
                    coords.append([lon, lat])
            except Exception:
                pass
        return coords

    def generate_linestring_coords(self, start_station: float, end_station: float,
                                    offset: float, combined_factor: float, transformer,
                                    spacing_ft: float = 25.0) -> List[List[float]]:
        """Generate coords for a line parallel to centerline at given offset."""
        coords = []
        station = start_station
        while station <= end_station:
            try:
                ground_e, ground_n = self.station_offset_to_grid_coords(station, offset)
                grid_e = ground_e / combined_factor
                grid_n = ground_n / combined_factor
                lon, lat = transformer.transform(grid_e, grid_n)
                elev = self.station_to_elevation(station)
                if elev is not None:
                    coords.append([lon, lat, elev])
                else:
                    coords.append([lon, lat])
            except Exception:
                pass
            station += spacing_ft
        if not coords or station - spacing_ft < end_station:
            try:
                ground_e, ground_n = self.station_offset_to_grid_coords(end_station, offset)
                grid_e = ground_e / combined_factor
                grid_n = ground_n / combined_factor
                lon, lat = transformer.transform(grid_e, grid_n)
                elev = self.station_to_elevation(end_station)
                if elev is not None:
                    coords.append([lon, lat, elev])
                else:
                    coords.append([lon, lat])
            except Exception:
                pass
        return coords
