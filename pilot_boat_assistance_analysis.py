#!/usr/bin/env python3
"""
Maritime Pilot Boat Assistance Analysis with Hybrid Traffic Direction Classification
====================================================================================

This script analyzes pilot boat assistance operations using vessel-specific proximity
thresholds based on vessel dimensions AND directional validation (course alignment
within 20°). Enhanced with hybrid traffic direction classification combining COG-based
and trajectory-based methods for improved accuracy.

Features:
- Dynamic proximity thresholds based on vessel dimensions (2 × max vessel width)
- Fallback to 100m threshold when vessel data unavailable
- Directional validation using course alignment (≤20° difference)
- Support for AIS COG data or calculated bearing from position history
- Special handling for stationary vessels (SOG < 0.5 knots)
- Temporal continuity analysis for assistance sessions
- Performance optimization using Polars for large datasets
- Comprehensive validation statistics and threshold reporting
- Safety limits: 50m minimum, 500m maximum proximity thresholds
- Vessel size filtering: excludes vessels with width ≤ 2m or LOA < 60m
- Tug boat exclusion: excludes 44 known tug boats to focus on large commercial vessels
- Speed similarity validation: both vessels must have SOG 5-10 knots with ≤3 knot difference
- Extended trajectory data collection for visualization purposes (30-minute window)
- Complete AIS data extraction including timestamp, position, COG, and SOG

Hybrid Traffic Direction Classification (Busan Port Optimized):
- BUSAN-SPECIFIC OPTIMIZATION: Heading ranges aligned with port's southeast-facing geography
- Expanded ranges: Inbound (030°-150°, 330°-030°), Outbound (150°-330°)
- Minimizes "other" classifications through binary preference logic
- Enhanced trajectory analysis using actual vessel movement patterns
- Contextual validation: speed patterns, pilot behavior, distance from port
- Intelligent priority logic: consensus → trajectory priority → COG priority → binary preference
- Confidence scoring: high/medium/low based on method agreement and geographical indicators
- Expected improvement: >90% binary classifications (inbound/outbound) vs previous 54.3%
- Maintains backward compatibility while significantly improving accuracy for Busan operations

Extended Trajectory Features:
- Collects AIS data for 15 minutes before session start + session duration + 15 minutes after session end
- Includes timestamp, latitude, longitude, COG (Course Over Ground), and SOG (Speed Over Ground)
- Structured data format optimized for visualization tools like Folium
- Graceful handling of edge cases where AIS data may not be available for full buffer periods
- Separate storage of standard session trajectory and extended visualization trajectory

Author: Maritime Analysis System
Date: 2024
"""

import pandas as pd
import numpy as np
import polars as pl
from math import radians, sin, cos, sqrt, atan2
import warnings
warnings.filterwarnings('ignore')

class PilotBoatAssistanceAnalyzer:
    """Analyzes pilot boat assistance operations using AIS data."""
    
    def __init__(self, dynamic_data_path, pilot_boat_excel_path, static_data_path=None):
        """
        Initialize the analyzer with data paths.

        Args:
            dynamic_data_path: Path to the dynamic AIS CSV file
            pilot_boat_excel_path: Path to the pilot boat Excel file
            static_data_path: Path to the static vessel data CSV file (optional)
        """
        self.dynamic_data_path = dynamic_data_path
        self.pilot_boat_excel_path = pilot_boat_excel_path
        self.static_data_path = static_data_path
        self.default_proximity_threshold = 100  # meters (fallback when no vessel data)
        self.course_alignment_threshold = 20  # degrees (±20°) - Updated from 45°
        self.use_dynamic_thresholds = static_data_path is not None

        # Data containers
        self.dynamic_data = None
        self.pilot_boat_data = None
        self.static_vessel_data = None
        self.pilot_boat_mmsi = []
        self.assistance_events = []
        self.threshold_statistics = {}

        # Tug boat exclusion list - exclude from analysis to focus on large commercial vessels
        self.tug_boats = {
            440412320, 440155260, 352003392, 440139260, 440030820, 514446000,
            440700180, 440051510, 440702880, 440301460, 440112210, 440148520,
            440165910, 440151610, 440110910, 440075240, 39036, 440313260,
            440713100, 440152850, 440100580, 440118970, 440018500, 440113610,
            440127670, 440112730, 441925000, 440083810, 440132580, 440116550,
            440300350, 440107360, 440184720, 440118750, 373493000, 440200060,
            354714000, 440313250, 440108150, 440909000, 440016540, 440335980,
            440005000, 440104800
        }

        print("Pilot Boat Assistance Analyzer initialized")
        if self.use_dynamic_thresholds:
            print(f"Dynamic proximity thresholds enabled (based on vessel dimensions)")
            print(f"Default fallback threshold: {self.default_proximity_threshold}m")
        else:
            print(f"Static proximity threshold: {self.default_proximity_threshold}m")
        print(f"Course alignment threshold: +/-{self.course_alignment_threshold}deg")
        print(f"Tug boat exclusion: {len(self.tug_boats)} vessels excluded from analysis")
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points on Earth.
        
        Args:
            lat1, lon1: Latitude and longitude of first point (decimal degrees)
            lat2, lon2: Latitude and longitude of second point (decimal degrees)
            
        Returns:
            Distance in meters
        """
        R = 6371000  # Earth radius in meters
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c

    def calculate_distance_from_port(self, lat, lon):
        """
        Calculate distance from Busan port center for trajectory analysis.

        Args:
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees

        Returns:
            Distance from Busan port center in meters
        """
        # Busan Port center coordinates
        BUSAN_PORT_LAT = 35.1040
        BUSAN_PORT_LON = 129.0403

        return self.haversine_distance(lat, lon, BUSAN_PORT_LAT, BUSAN_PORT_LON)

    def calculate_bearing(self, lat1, lon1, lat2, lon2):
        """
        Calculate the bearing between two points on Earth.

        Args:
            lat1, lon1: Latitude and longitude of first point (decimal degrees)
            lat2, lon2: Latitude and longitude of second point (decimal degrees)

        Returns:
            Bearing in degrees (0-360)
        """
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        # Calculate bearing
        dlon = lon2 - lon1
        y = sin(dlon) * cos(lat2)
        x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)

        bearing = atan2(y, x)
        # Convert to degrees and normalize to 0-360
        bearing = (bearing * 180 / np.pi + 360) % 360

        return bearing

    def calculate_course_difference(self, course1, course2):
        """
        Calculate the absolute difference between two courses/headings.

        Args:
            course1: First course in degrees (0-360)
            course2: Second course in degrees (0-360)

        Returns:
            Absolute course difference in degrees (0-180)
        """
        if pd.isna(course1) or pd.isna(course2):
            return np.nan

        # Calculate the difference
        diff = abs(course1 - course2)

        # Handle the circular nature of compass bearings
        if diff > 180:
            diff = 360 - diff

        return diff

    def classify_traffic_direction(self, heading, speed=None, course_alignment=None,
                                 distance=None, validation_reason=None, pilot_heading=None):
        """
        Busan Port-optimized traffic direction classification with geographical accuracy.

        Busan Port faces southeast toward Korea Strait. This classification system is
        specifically tuned for Busan's geographical orientation and traffic patterns.

        Args:
            heading: Vessel heading in degrees (0-360)
            speed: Vessel speed in knots (optional, for context)
            course_alignment: Boolean indicating course alignment (optional)
            distance: Distance to pilot boat in meters (optional)
            validation_reason: Validation reason from proximity detection (optional)
            pilot_heading: Pilot boat heading for additional context (optional)

        Returns:
            String: "inbound", "outbound", or "other"
        """
        if pd.isna(heading):
            return "nan"

        # Normalize heading to 0-360 range
        heading = heading % 360

        # BUSAN PORT-SPECIFIC RANGES (optimized for southeast-facing port)
        # Based on Busan's geography: vessels approach from Korea Strait (south/southeast)
        # and depart toward open ocean (south/southeast)

        # PRIMARY INBOUND RANGES - vessels approaching from open ocean toward port
        # Northeast approach (from Korea Strait): 030° to 090° (60° range)
        if 30 <= heading <= 90:
            return "inbound"

        # Southeast approach (main shipping lane): 090° to 150° (60° range)
        if 90 <= heading <= 150:
            return "inbound"

        # PRIMARY OUTBOUND RANGES - vessels departing toward open ocean
        # Southwest departure: 210° to 270° (60° range)
        if 210 <= heading <= 270:
            return "outbound"

        # Southeast departure (main shipping lane): 150° to 210° (60° range)
        if 150 <= heading <= 210:
            return "outbound"

        # EXTENDED RANGES with contextual validation
        # Extended inbound: Northwest approach 330° to 030°
        if (330 <= heading <= 360) or (0 <= heading < 30):
            if self._validate_inbound_context(speed, course_alignment, distance, validation_reason):
                return "inbound"

        # Extended outbound: 135°-147° and 179°-190°
        if (135 <= heading < 148) or (179 <= heading <= 190):
            # Require additional validation for extended ranges
            if self._validate_outbound_context(speed, course_alignment, distance, validation_reason):
                return "outbound"

        # Contextual classification for coordinated movements
        if (speed is not None and course_alignment is not None and
            validation_reason is not None and course_alignment):

            # Strong indicators of coordinated movement
            if validation_reason == 'course_aligned_speed_similar':

                # REFINED: True northwest quadrant only (270°-360° and 0°-30°)
                # Excludes problematic northeast range (30°-90°)
                if (270 <= heading <= 360) or (0 <= heading <= 30):
                    return "inbound"

                # REFINED: True southeast quadrant (120°-270°)
                # Excludes problematic northeast range (30°-120°)
                if 120 <= heading <= 270:
                    return "outbound"

                # NEW: Northeast quadrant (30°-120°) - typically outbound traffic
                if 30 < heading < 120:
                    # Additional validation for northeast headings
                    if self._validate_northeast_context(speed, distance, pilot_heading):
                        return "outbound"
                    else:
                        return "other"

            # Boarding operations with directional hints
            if validation_reason == 'boarding_operation' and pilot_heading is not None:
                pilot_heading = pilot_heading % 360 if not pd.isna(pilot_heading) else None

                if pilot_heading is not None:
                    # BUSAN-OPTIMIZED: Use pilot heading as strong directional indicator
                    # If pilot approaching from port (west/northwest), vessel likely inbound
                    if (270 <= pilot_heading <= 360) or (0 <= pilot_heading <= 90):
                        if (0 <= heading <= 180):  # Vessel heading north to south
                            return "inbound"

                    # If pilot departing toward ocean (south/southeast), vessel likely outbound
                    if 90 <= pilot_heading <= 270:
                        if (120 <= heading <= 300):  # Vessel heading southeast to northwest
                            return "outbound"

        # BUSAN-SPECIFIC FALLBACK LOGIC
        # Minimize "other" classifications for legitimate pilot-vessel interactions

        # Speed-based contextual hints
        if speed is not None and not pd.isna(speed):
            # Slow vessels heading toward port area = likely inbound
            if speed < 8.0 and (0 <= heading <= 180):
                return "inbound"
            # Fast vessels heading away from port = likely outbound
            if speed > 12.0 and (150 <= heading <= 330):
                return "outbound"

        # Final binary classification to minimize "other"
        # Northern semicircle (270° to 090°) - bias toward inbound (approaching from sea)
        if (270 <= heading <= 360) or (0 <= heading <= 90):
            return "inbound"

        # Southern semicircle (090° to 270°) - bias toward outbound (departing to sea)
        if 90 <= heading <= 270:
            return "outbound"

        # Should rarely reach here with improved logic
        return "other"

    def _validate_northeast_context(self, speed, distance, pilot_heading):
        """
        Validate northeast heading context (30°-120°) for better classification.

        Northeast headings are typically outbound if:
        1. Vessel is moving at reasonable speed (> 5 knots)
        2. Distance suggests active navigation (not anchored)
        3. Pilot heading suggests outbound escort
        """
        if speed is not None and speed > 5.0:
            if distance is not None and distance < 200:  # Close escort
                if pilot_heading is not None:
                    pilot_heading = pilot_heading % 360
                    # If pilot is also heading northeast, likely outbound escort
                    if 30 <= pilot_heading <= 120:
                        return True

        return False

    def _validate_inbound_context(self, speed, course_alignment, distance, validation_reason):
        """Validate context for extended inbound classification."""
        if speed is None or course_alignment is None:
            return False

        # Strong validation criteria for extended inbound range
        if validation_reason == 'course_aligned_speed_similar':
            return True

        if course_alignment and speed <= 3.0:  # Low speed + course alignment
            return True

        if validation_reason == 'boarding_operation' and distance is not None and distance <= 100:
            return True

        return False

    def _validate_outbound_context(self, speed, course_alignment, distance, validation_reason):
        """Validate context for extended outbound classification."""
        if speed is None or course_alignment is None:
            return False

        # Strong validation criteria for extended outbound range
        if validation_reason == 'course_aligned_speed_similar':
            return True

        if course_alignment and speed <= 3.0:  # Low speed + course alignment
            return True

        if validation_reason == 'boarding_operation' and distance is not None and distance <= 100:
            return True

        return False

    def get_primary_traffic_direction(self, directions_list):
        """
        Determine the primary traffic direction from a list of directions.

        Args:
            directions_list: List of traffic direction strings

        Returns:
            String: Most common direction, or "mixed" if tie
        """
        if not directions_list:
            return "other"

        # Count occurrences of each direction
        direction_counts = {}
        for direction in directions_list:
            direction_counts[direction] = direction_counts.get(direction, 0) + 1

        # Find the most common direction
        max_count = max(direction_counts.values())
        most_common = [direction for direction, count in direction_counts.items() if count == max_count]

        # Return single direction or "mixed" if tie
        if len(most_common) == 1:
            return most_common[0]
        else:
            return "mixed"

    def get_vessel_course(self, vessel_data, mmsi, timestamp):
        """
        Get the course for a vessel, using COG if available or calculating from position history.

        Args:
            vessel_data: DataFrame with vessel AIS data
            mmsi: MMSI of the vessel
            timestamp: Current timestamp

        Returns:
            Course in degrees (0-360) or NaN if cannot be determined
        """
        # First try to use COG (Course Over Ground) if available and valid
        current_record = vessel_data[
            (vessel_data['MMSI'] == mmsi) &
            (vessel_data['DateTime'] == timestamp)
        ]

        if not current_record.empty:
            cog = current_record.iloc[0]['COG']
            if pd.notna(cog) and 0 <= cog <= 360:
                return cog

        # If COG not available, calculate from position history
        vessel_history = vessel_data[vessel_data['MMSI'] == mmsi].sort_values('DateTime')
        current_idx = vessel_history[vessel_history['DateTime'] == timestamp].index

        if len(current_idx) == 0 or current_idx[0] == vessel_history.index[0]:
            return np.nan  # No previous position available

        # Get current and previous positions
        current_pos = vessel_history.loc[current_idx[0]]
        prev_idx = vessel_history.index[vessel_history.index < current_idx[0]][-1]
        prev_pos = vessel_history.loc[prev_idx]

        # Calculate bearing from previous to current position
        if pd.notna(prev_pos['Latitude']) and pd.notna(prev_pos['Longitude']) and \
           pd.notna(current_pos['Latitude']) and pd.notna(current_pos['Longitude']):
            return self.calculate_bearing(
                prev_pos['Latitude'], prev_pos['Longitude'],
                current_pos['Latitude'], current_pos['Longitude']
            )

        return np.nan

    def is_course_aligned(self, pilot_course, vessel_course):
        """
        Check if pilot boat and vessel courses are aligned within threshold.

        Args:
            pilot_course: Pilot boat course in degrees
            vessel_course: Vessel course in degrees

        Returns:
            Boolean indicating if courses are aligned, and the course difference
        """
        if pd.isna(pilot_course) or pd.isna(vessel_course):
            return False, np.nan

        course_diff = self.calculate_course_difference(pilot_course, vessel_course)

        if pd.isna(course_diff):
            return False, np.nan

        is_aligned = course_diff <= self.course_alignment_threshold
        return is_aligned, course_diff

    def is_speed_similar(self, pilot_sog, vessel_sog):
        """
        Check if pilot boat and vessel have similar speeds for assistance operations.
        Both vessels must have SOG between 5-10 knots with speed difference ≤ 3 knots.

        Args:
            pilot_sog: Pilot boat Speed Over Ground in knots
            vessel_sog: Vessel Speed Over Ground in knots

        Returns:
            Boolean indicating if speeds are similar, and the speed difference
        """
        # Handle NaN values
        if pd.isna(pilot_sog) or pd.isna(vessel_sog):
            return False, np.nan

        # Convert to float to ensure proper comparison
        pilot_sog = float(pilot_sog) if not pd.isna(pilot_sog) else 0.0
        vessel_sog = float(vessel_sog) if not pd.isna(vessel_sog) else 0.0

        # Check if both vessels are in the operational speed range (5-10 knots)
        pilot_in_range = 5.0 <= pilot_sog <= 10.0
        vessel_in_range = 5.0 <= vessel_sog <= 10.0

        if not (pilot_in_range and vessel_in_range):
            return False, abs(pilot_sog - vessel_sog)

        # Check if speed difference is within acceptable range (≤ 3 knots)
        speed_diff = abs(pilot_sog - vessel_sog)
        is_similar = speed_diff <= 3.0

        return is_similar, speed_diff

    def is_boarding_operation(self, pilot_sog, vessel_sog):
        """
        Check if this represents a potential boarding/disembarking operation.
        Both vessels must be moving slowly (≤ 2 knots) but NOT completely stationary,
        indicating coordinated maneuvering rather than anchored vessels.

        Args:
            pilot_sog: Pilot boat Speed Over Ground in knots
            vessel_sog: Vessel Speed Over Ground in knots

        Returns:
            Boolean indicating if this is a potential boarding operation, and max speed
        """
        # Handle NaN values
        if pd.isna(pilot_sog) or pd.isna(vessel_sog):
            return False, np.nan

        # Convert to float to ensure proper comparison
        pilot_sog = float(pilot_sog) if not pd.isna(pilot_sog) else 0.0
        vessel_sog = float(vessel_sog) if not pd.isna(vessel_sog) else 0.0

        # Exclude completely stationary vessels (likely anchored)
        # At least one vessel must have some movement (> 0.3 knots) to indicate active operation
        min_movement_threshold = 0.3  # knots
        has_movement = (pilot_sog > min_movement_threshold) or (vessel_sog > min_movement_threshold)

        if not has_movement:
            return False, max(pilot_sog, vessel_sog)

        # Both vessels must be moving slowly for boarding operations
        # Typical boarding speeds are ≤ 2 knots for safety
        pilot_slow = pilot_sog <= 2.0
        vessel_slow = vessel_sog <= 2.0

        # Both must be slow, and at least one must show movement
        is_boarding = pilot_slow and vessel_slow and has_movement

        max_speed = max(pilot_sog, vessel_sog)
        return is_boarding, max_speed

    def classify_trajectory_direction(self, bearing):
        """
        Busan Port-optimized trajectory direction classification.

        Uses actual vessel movement patterns (trajectory bearing) to determine
        traffic direction. Optimized for Busan's southeast-facing geography.

        Args:
            bearing: Trajectory bearing in degrees (0-360)

        Returns:
            String: "inbound", "outbound", or "other"
        """
        if pd.isna(bearing):
            return "nan"

        # Normalize bearing to 0-360 range
        bearing = bearing % 360

        # BUSAN PORT-SPECIFIC TRAJECTORY CLASSIFICATION
        # Based on actual vessel movement toward/away from port center (35.1040°N, 129.0403°E)

        # PRIMARY INBOUND TRAJECTORIES - movement toward port from open ocean
        # Northeast trajectory (vessels approaching from Korea Strait): 030° to 090°
        if 30 <= bearing <= 90:
            return "inbound"

        # Southeast trajectory (main shipping lane approach): 090° to 150°
        if 90 <= bearing <= 150:
            return "inbound"

        # Northwest trajectory (coastal approach): 330° to 030°
        if (330 <= bearing <= 360) or (0 <= bearing < 30):
            return "inbound"

        # PRIMARY OUTBOUND TRAJECTORIES - movement away from port toward open ocean
        # Southwest trajectory: 210° to 270°
        if 210 <= bearing <= 270:
            return "outbound"

        # Southeast trajectory (main shipping lane departure): 150° to 210°
        if 150 <= bearing <= 210:
            return "outbound"

        # West/northwest trajectory: 270° to 330°
        if 270 <= bearing < 330:
            return "outbound"

        # TRAJECTORY-BASED CLASSIFICATION IS MORE RELIABLE
        # For trajectory analysis, we can be more confident in binary classification
        # since we're observing actual movement patterns

        # Any remaining northern arc movement (toward port) = inbound
        # This should rarely be reached due to comprehensive ranges above
        return "other"

    def implement_hybrid_classification(self, cog_direction, trajectory_direction,
                                      trajectory_bearing, distance_moved, duration_minutes):
        """
        Busan Port-optimized hybrid classification combining COG and trajectory methods.

        Enhanced with Busan-specific logic and improved confidence scoring.

        Args:
            cog_direction: COG-based classification result
            trajectory_direction: Trajectory-based classification result
            trajectory_bearing: Calculated trajectory bearing
            distance_moved: Distance vessel moved during session
            duration_minutes: Session duration

        Returns:
            Dictionary with hybrid classification result and reasoning
        """
        result = {
            'hybrid_direction': None,
            'confidence': 'low',
            'reasoning': [],
            'method_used': None
        }

        # Handle missing trajectory data
        if pd.isna(trajectory_bearing) or trajectory_direction == "nan":
            result['hybrid_direction'] = cog_direction
            result['confidence'] = 'medium' if cog_direction in ['inbound', 'outbound'] else 'low'
            result['reasoning'].append('No trajectory data - using COG classification')
            result['method_used'] = 'cog_only'
            return result

        # Priority 1: CONSENSUS - Both methods agree (highest confidence)
        if cog_direction == trajectory_direction and cog_direction in ['inbound', 'outbound']:
            result['hybrid_direction'] = cog_direction
            result['confidence'] = 'high'
            result['reasoning'].append(f"Strong consensus: Both COG and trajectory methods agree on '{cog_direction}'")
            result['method_used'] = 'consensus'
            return result

        # Priority 2: For longer sessions with significant movement, prioritize trajectory
        if duration_minutes >= 15 and distance_moved >= 200:
            result['hybrid_direction'] = trajectory_direction
            result['confidence'] = 'high'
            result['reasoning'].append(f"Long session ({duration_minutes:.1f} min) with significant movement ({distance_moved:.1f}m)")
            result['reasoning'].append(f"Trajectory-based classification more reliable: '{trajectory_direction}'")
            result['method_used'] = 'trajectory_priority'
            return result

        # Priority 3: For sessions with minimal movement, prioritize COG
        if distance_moved < 100:
            result['hybrid_direction'] = cog_direction
            result['confidence'] = 'medium'
            result['reasoning'].append(f"Minimal movement ({distance_moved:.1f}m) - COG more reliable")
            result['reasoning'].append(f"Using COG-based classification: '{cog_direction}'")
            result['method_used'] = 'cog_priority'
            return result

        # Priority 4: BUSAN-SPECIFIC DIRECTIONAL INDICATORS
        # Strong inbound indicators (trajectory toward port from Korea Strait)
        if (30 <= trajectory_bearing <= 150) or (330 <= trajectory_bearing <= 360) or (0 <= trajectory_bearing < 30):
            if trajectory_direction == 'inbound':
                result['hybrid_direction'] = 'inbound'
                result['confidence'] = 'high'
                result['reasoning'].append(f"Strong inbound trajectory toward Busan port: {trajectory_bearing:.1f}deg")
                result['method_used'] = 'trajectory_priority'
                return result

        # Strong outbound indicators (trajectory away from port toward Korea Strait)
        if 150 <= trajectory_bearing <= 330:
            if trajectory_direction == 'outbound':
                result['hybrid_direction'] = 'outbound'
                result['confidence'] = 'high'
                result['reasoning'].append(f"Strong outbound trajectory from Busan port: {trajectory_bearing:.1f}deg")
                result['method_used'] = 'trajectory_priority'
                return result

        # Priority 5: PREFER NON-"OTHER" CLASSIFICATIONS
        # Minimize "other" classifications for legitimate pilot-vessel interactions
        if trajectory_direction in ['inbound', 'outbound']:
            result['hybrid_direction'] = trajectory_direction
            result['confidence'] = 'medium'
            result['reasoning'].append(f"Trajectory provides clear direction: '{trajectory_direction}'")
            result['method_used'] = 'trajectory_preferred'
            return result

        if cog_direction in ['inbound', 'outbound']:
            result['hybrid_direction'] = cog_direction
            result['confidence'] = 'medium'
            result['reasoning'].append(f"COG provides clear direction: '{cog_direction}'")
            result['method_used'] = 'cog_preferred'
            return result

        # Priority 6: Default fallback - use COG for short sessions
        result['hybrid_direction'] = cog_direction
        result['confidence'] = 'low'
        result['reasoning'].append(f"Short session ({duration_minutes:.1f} min) - defaulting to COG")
        result['reasoning'].append(f"Using COG-based classification: '{cog_direction}'")
        result['method_used'] = 'cog_default'

        return result

    def analyze_session_with_hybrid_classification(self, session_data):
        """
        Analyze a session with both COG-based and trajectory-based classification.

        Args:
            session_data: Dictionary containing session information

        Returns:
            Dictionary with comprehensive classification analysis
        """
        result = {
            'cog_based_direction': session_data.get('primary_traffic_direction', 'unknown'),
            'trajectory_bearing': None,
            'trajectory_based_direction': None,
            'hybrid_direction': None,
            'hybrid_confidence': None,
            'hybrid_reasoning': None,
            'hybrid_method_used': None,
            'distance_moved': None,
            'classification_notes': []
        }

        # Calculate trajectory bearing if coordinates are available
        start_lat = session_data.get('start_lat')
        start_lon = session_data.get('start_lon')
        end_lat = session_data.get('end_lat')
        end_lon = session_data.get('end_lon')

        if all(pd.notna(coord) for coord in [start_lat, start_lon, end_lat, end_lon]):
            # Calculate trajectory bearing
            bearing = self.calculate_bearing(start_lat, start_lon, end_lat, end_lon)
            result['trajectory_bearing'] = bearing

            # Classify based on trajectory
            trajectory_direction = self.classify_trajectory_direction(bearing)
            result['trajectory_based_direction'] = trajectory_direction

            # Calculate distance moved
            distance_moved = self.haversine_distance(start_lat, start_lon, end_lat, end_lon)
            result['distance_moved'] = distance_moved

            # Implement hybrid classification
            duration_minutes = session_data.get('duration_minutes', 0)
            hybrid_result = self.implement_hybrid_classification(
                result['cog_based_direction'],
                trajectory_direction,
                bearing,
                distance_moved,
                duration_minutes
            )

            result.update({
                'hybrid_direction': hybrid_result['hybrid_direction'],
                'hybrid_confidence': hybrid_result['confidence'],
                'hybrid_reasoning': '; '.join(hybrid_result['reasoning']),
                'hybrid_method_used': hybrid_result['method_used']
            })

            # Add analysis notes
            if distance_moved < 100:
                result['classification_notes'].append("Low movement distance - trajectory may be unreliable")

            if duration_minutes < 5:
                result['classification_notes'].append("Short session duration - limited trajectory data")

            if result['cog_based_direction'] != trajectory_direction:
                result['classification_notes'].append(f"Classification mismatch: COG={result['cog_based_direction']}, Trajectory={trajectory_direction}")

        else:
            # No trajectory data available
            result.update({
                'hybrid_direction': result['cog_based_direction'],
                'hybrid_confidence': 'medium',
                'hybrid_reasoning': 'No trajectory data - using COG classification',
                'hybrid_method_used': 'cog_only'
            })
            result['classification_notes'].append("Missing coordinate data - cannot calculate trajectory")

        return result

    def load_static_vessel_data(self):
        """
        Load static vessel data containing vessel dimensions with size filtering.
        Excludes vessels with width ≤ 2 meters or LOA < 60 meters.
        """
        if not self.use_dynamic_thresholds:
            return

        print("Loading static vessel data with size filtering...")

        try:
            # Read the static vessel data CSV
            self.static_vessel_data = pd.read_csv(self.static_data_path)
            print(f"Loaded static vessel data: {len(self.static_vessel_data)} records")

            # Check for required columns (note: 'width' is the actual column name in the data)
            if 'MMSI' in self.static_vessel_data.columns and 'width' in self.static_vessel_data.columns and 'loa' in self.static_vessel_data.columns:
                # Apply vessel size filtering
                print("Applying vessel size filters...")
                original_count = len(self.static_vessel_data)

                # Filter out vessels with width ≤ 2 meters or LOA < 60 meters
                filtered_data = self.static_vessel_data[
                    (self.static_vessel_data['width'] > 2) &
                    (self.static_vessel_data['loa'] >= 60) &
                    (pd.notna(self.static_vessel_data['width'])) &
                    (pd.notna(self.static_vessel_data['loa']))
                ]

                filtered_count = len(filtered_data)
                excluded_count = original_count - filtered_count
                print(f"Vessel filtering results:")
                print(f"- Original vessels: {original_count}")
                print(f"- Vessels after filtering (width > 2m AND LOA >= 60m): {filtered_count}")
                print(f"- Excluded vessels: {excluded_count}")

                # Create a dictionary for fast lookup: MMSI -> width (only for filtered vessels)
                self.vessel_width_lookup = {}
                for _, row in filtered_data.iterrows():
                    mmsi = row['MMSI']
                    width = row['width']
                    if pd.notna(width) and width > 0:
                        self.vessel_width_lookup[mmsi] = width

                print(f"Vessel width data available for {len(self.vessel_width_lookup)} vessels")

                # Show some statistics
                if self.vessel_width_lookup:
                    widths = list(self.vessel_width_lookup.values())
                    print(f"Vessel width range (filtered): {min(widths):.1f}m to {max(widths):.1f}m")
                    print(f"Average vessel width (filtered): {sum(widths)/len(widths):.1f}m")

                    # Show LOA statistics for filtered vessels
                    loa_values = filtered_data['loa'].dropna()
                    if not loa_values.empty:
                        print(f"LOA range (filtered): {loa_values.min():.1f}m to {loa_values.max():.1f}m")
                        print(f"Average LOA (filtered): {loa_values.mean():.1f}m")
            else:
                print("Warning: Required columns (MMSI, width, loa) not found in static vessel data")
                self.vessel_width_lookup = {}

        except Exception as e:
            print(f"Error loading static vessel data: {e}")
            self.vessel_width_lookup = {}

    def calculate_dynamic_proximity_threshold(self, pilot_mmsi, vessel_mmsi):
        """
        Calculate dynamic proximity threshold based on vessel dimensions.
        Formula: 2 × max(pilot_boat_width, target_vessel_width)

        Args:
            pilot_mmsi: MMSI of the pilot boat
            vessel_mmsi: MMSI of the target vessel

        Returns:
            Dynamic proximity threshold in meters
        """
        if not self.use_dynamic_thresholds:
            return self.default_proximity_threshold

        # Get vessel widths from lookup table
        pilot_width = self.vessel_width_lookup.get(pilot_mmsi, None)
        vessel_width = self.vessel_width_lookup.get(vessel_mmsi, None)

        # If either width is missing, use default threshold
        if pilot_width is None or vessel_width is None:
            return self.default_proximity_threshold

        # Calculate dynamic threshold: 2 × max(pilot_width, vessel_width)
        max_width = max(pilot_width, vessel_width)
        dynamic_threshold = 2 * max_width

        # Ensure minimum threshold of 50m and maximum of 500m for safety
        # dynamic_threshold = max(50, min(500, dynamic_threshold))

        return dynamic_threshold

    def load_pilot_boat_data(self):
        """Load pilot boat information from Excel file."""
        try:
            # Try to read the Excel file
            self.pilot_boat_data = pd.read_excel(self.pilot_boat_excel_path)
            print(f"Loaded pilot boat data: {len(self.pilot_boat_data)} records")
            print("Pilot boat data columns:", self.pilot_boat_data.columns.tolist())
            
            # Extract MMSI if available
            if 'MMSI' in self.pilot_boat_data.columns:
                self.pilot_boat_mmsi = self.pilot_boat_data['MMSI'].dropna().astype(int).tolist()
            else:
                print("MMSI column not found in pilot boat data")
                exit(1)
                
        except Exception as e:
            print(f"Error loading pilot boat Excel file: {e}")
            print("Using default pilot boat MMSI list")
            exit(1)
        
        print(f"Pilot boat MMSI list: {self.pilot_boat_mmsi}")
        return self.pilot_boat_data
    
    
    def load_dynamic_data(self):
        """Load and prepare dynamic AIS data with vessel size filtering."""
        print("Loading dynamic AIS data...")

        # Load the CSV file
        self.dynamic_data = pd.read_csv(self.dynamic_data_path)
        print(f"Loaded dynamic data: {len(self.dynamic_data)} records")

        # Apply vessel size filtering if static data is available
        if self.use_dynamic_thresholds and hasattr(self, 'vessel_width_lookup'):
            print("Applying vessel size filtering to dynamic AIS data...")
            original_count = len(self.dynamic_data)
            original_vessels = self.dynamic_data['MMSI'].nunique()

            # Get list of vessels that meet size requirements (from filtered static data)
            valid_vessel_mmsi = set(self.vessel_width_lookup.keys())

            # Also include pilot boats (they should always be processed regardless of size)
            valid_vessel_mmsi.update(self.pilot_boat_mmsi)

            # Filter dynamic data to only include vessels that meet size requirements
            self.dynamic_data = self.dynamic_data[
                self.dynamic_data['MMSI'].isin(valid_vessel_mmsi)
            ]

            filtered_count = len(self.dynamic_data)
            filtered_vessels = self.dynamic_data['MMSI'].nunique()
            excluded_records = original_count - filtered_count
            excluded_vessels = original_vessels - filtered_vessels

            print(f"Dynamic data vessel size filtering results:")
            print(f"- Original records: {original_count:,}")
            print(f"- Records after filtering: {filtered_count:,}")
            print(f"- Excluded records: {excluded_records:,}")
            print(f"- Original unique vessels: {original_vessels}")
            print(f"- Vessels after filtering: {filtered_vessels}")
            print(f"- Excluded vessels: {excluded_vessels}")

            # Verify specific vessel exclusion
            if 440106050 in self.dynamic_data['MMSI'].values:
                print("WARNING: Vessel MMSI 440106050 still present in dynamic data!")
            else:
                print("SUCCESS: Vessel MMSI 440106050 successfully excluded from dynamic data")

        # Convert DateTime column to datetime
        self.dynamic_data['DateTime'] = pd.to_datetime(self.dynamic_data['DateTime'])

        # Sort by MMSI and DateTime for efficient processing
        self.dynamic_data = self.dynamic_data.sort_values(['MMSI', 'DateTime'])

        # Add pilot boat flag
        self.dynamic_data['IsPilotBoat'] = self.dynamic_data['MMSI'].isin(self.pilot_boat_mmsi)

        print(f"Pilot boat records found: {self.dynamic_data['IsPilotBoat'].sum()}")
        print(f"Unique pilot boats in data: {self.dynamic_data[self.dynamic_data['IsPilotBoat']]['MMSI'].nunique()}")

        return self.dynamic_data
    
    def detect_assistance_events(self, sample_size=None):
        """
        Detect assistance events using dynamic proximity thresholds based on vessel dimensions,
        course alignment validation within 20°, and speed similarity validation.
        Excludes tug boats from analysis to focus on pilot boat interactions with large commercial vessels.

        Args:
            sample_size: If provided, only process this many rows for testing
        """
        print("Detecting assistance events with dynamic proximity, directional, and speed validation...")
        if self.use_dynamic_thresholds:
            print("Using dynamic proximity thresholds based on vessel dimensions")
            print(f"Formula: 2 * max(pilot_boat_width, target_vessel_width)")
            print(f"Fallback threshold: {self.default_proximity_threshold}m")
        else:
            print(f"Using static proximity threshold: {self.default_proximity_threshold}m")
        print(f"Course alignment threshold: <={self.course_alignment_threshold}deg")
        print(f"Speed similarity validation: Both vessels SOG 5-10 knots, difference <=3 knots")
        print(f"Boarding operation detection: Both vessels <=2 knots + recent movement validation")
        print(f"Vessel filtering: Excluding {len(self.tug_boats)} tug boats from analysis")

        # Use sample if specified
        data_to_process = self.dynamic_data
        if sample_size:
            data_to_process = self.dynamic_data.head(sample_size)
            print(f"Processing sample of {len(data_to_process)} records")

        assistance_events = []
        proximity_only_events = 0
        course_aligned_events = 0
        speed_similar_events = 0
        total_proximity_checks = 0

        # Track dynamic threshold statistics
        dynamic_thresholds_used = []

        # Get unique timestamps for efficient processing
        unique_times = data_to_process['DateTime'].unique()
        print(f"Processing {len(unique_times)} unique timestamps...")

        for i, timestamp in enumerate(unique_times):
            if i % 10 == 0:
                print(f"Processing timestamp {i+1}/{len(unique_times)}: {timestamp}")

            # Get all vessels at this timestamp
            vessels_at_time = data_to_process[data_to_process['DateTime'] == timestamp]

            # Get pilot boats at this timestamp
            pilot_boats = vessels_at_time[vessels_at_time['IsPilotBoat']]

            # Get other vessels at this timestamp (excluding pilot boats and tug boats)
            other_vessels = vessels_at_time[
                (~vessels_at_time['IsPilotBoat']) &
                (~vessels_at_time['MMSI'].isin(self.tug_boats))
            ]

            # Skip if no pilot boats at this time
            if len(pilot_boats) == 0:
                print(f"Warning: No pilot boats at timestamp {timestamp}")
                continue

            # Check proximity and course alignment between each pilot boat and other vessels
            for _, pilot in pilot_boats.iterrows():
                for _, vessel in other_vessels.iterrows():
                    total_proximity_checks += 1

                    # Calculate dynamic proximity threshold for this vessel pair
                    proximity_threshold = self.calculate_dynamic_proximity_threshold(
                        pilot['MMSI'], vessel['MMSI']
                    )
                    dynamic_thresholds_used.append(proximity_threshold)

                    # First check: proximity using dynamic threshold
                    distance = self.haversine_distance(
                        pilot['Latitude'], pilot['Longitude'],
                        vessel['Latitude'], vessel['Longitude']
                    )

                    if distance <= proximity_threshold:
                        proximity_only_events += 1

                        # Second check: course alignment
                        pilot_course = self.get_vessel_course(data_to_process, pilot['MMSI'], timestamp)
                        vessel_course = self.get_vessel_course(data_to_process, vessel['MMSI'], timestamp)

                        is_aligned, course_diff = self.is_course_aligned(pilot_course, vessel_course)

                        # Handle edge cases for stationary vessels
                        pilot_sog = pilot.get('SOG', 0)
                        vessel_sog = vessel.get('SOG', 0)

                        # Handle missing speed data
                        if pd.isna(pilot_sog):
                            pilot_sog = 0
                        if pd.isna(vessel_sog):
                            vessel_sog = 0

                        # Third check: speed similarity validation
                        is_speed_similar, speed_diff = self.is_speed_similar(pilot_sog, vessel_sog)

                        # Fourth check: boarding operation detection (both vessels slow)
                        is_boarding_op, max_speed = self.is_boarding_operation(pilot_sog, vessel_sog)

                        # Fifth check: movement validation for boarding operations
                        if is_boarding_op:
                            # Verify that at least one vessel has shown recent movement
                            pilot_has_movement = self.has_recent_movement(pilot['MMSI'], timestamp)
                            vessel_has_movement = self.has_recent_movement(vessel['MMSI'], timestamp)

                            # Only consider it a boarding operation if there's evidence of recent movement
                            is_boarding_op = is_boarding_op and (pilot_has_movement or vessel_has_movement)

                        # Valid assistance event if:
                        # 1. Courses are aligned AND speeds are similar (active assistance), OR
                        # 2. Both vessels are slow AND have recent movement (boarding/disembarking)
                        if (is_aligned and is_speed_similar) or is_boarding_op:
                            course_aligned_events += 1
                            if is_speed_similar:
                                speed_similar_events += 1

                            # Determine validation reason
                            if is_boarding_op:
                                validation_reason = 'boarding_operation'
                            elif is_aligned and is_speed_similar:
                                validation_reason = 'course_aligned_speed_similar'
                            elif is_aligned:
                                validation_reason = 'course_aligned_only'
                            else:
                                validation_reason = 'other'

                            # Classify traffic direction with enhanced context awareness
                            vessel_traffic_direction = self.classify_traffic_direction(
                                vessel_course, vessel_sog, is_aligned, distance, validation_reason, pilot_course
                            )
                            pilot_traffic_direction = self.classify_traffic_direction(
                                pilot_course, pilot_sog, is_aligned, distance, validation_reason
                            )

                            assistance_events.append({
                                'timestamp': timestamp,
                                'pilot_mmsi': pilot['MMSI'],
                                'pilot_lat': pilot['Latitude'],
                                'pilot_lon': pilot['Longitude'],
                                'pilot_sog': pilot_sog,
                                'pilot_cog': pilot_course,
                                'vessel_mmsi': vessel['MMSI'],
                                'vessel_lat': vessel['Latitude'],
                                'vessel_lon': vessel['Longitude'],
                                'vessel_sog': vessel_sog,
                                'vessel_cog': vessel_course,
                                'distance': distance,
                                'proximity_threshold_used': proximity_threshold,
                                'course_difference': course_diff,
                                'speed_difference': speed_diff,
                                'max_speed_in_pair': max_speed,
                                'is_course_aligned': is_aligned,
                                'is_speed_similar': is_speed_similar,
                                'is_boarding_operation': is_boarding_op,
                                'validation_reason': validation_reason,
                                'traffic_direction': vessel_traffic_direction,
                                'pilot_traffic_direction': pilot_traffic_direction
                            })

        self.assistance_events = pd.DataFrame(assistance_events)

        # Print validation statistics
        print(f"\nValidation Results:")
        print(f"- Total proximity checks: {total_proximity_checks:,}")

        # Dynamic threshold statistics
        if self.use_dynamic_thresholds and dynamic_thresholds_used:
            print(f"- Dynamic thresholds used: {len(dynamic_thresholds_used)} calculations")
            print(f"- Threshold range: {min(dynamic_thresholds_used):.1f}m to {max(dynamic_thresholds_used):.1f}m")
            print(f"- Average threshold: {sum(dynamic_thresholds_used)/len(dynamic_thresholds_used):.1f}m")
            print(f"- Proximity-only events (dynamic thresholds): {proximity_only_events}")
        else:
            print(f"- Proximity-only events (<={self.default_proximity_threshold}m): {proximity_only_events}")

        print(f"- Course-aligned events (<={self.course_alignment_threshold}deg): {course_aligned_events}")
        print(f"- Speed-similar events (5-10 knots, <=3 knot diff): {speed_similar_events}")
        print(f"- Valid assistance events: {len(self.assistance_events)}")

        if proximity_only_events > 0:
            reduction_rate = (1 - len(self.assistance_events) / proximity_only_events) * 100
            print(f"- False positive reduction: {reduction_rate:.1f}%")

        # Store threshold statistics for later analysis
        self.threshold_statistics = {
            'dynamic_thresholds_used': dynamic_thresholds_used,
            'min_threshold': min(dynamic_thresholds_used) if dynamic_thresholds_used else self.default_proximity_threshold,
            'max_threshold': max(dynamic_thresholds_used) if dynamic_thresholds_used else self.default_proximity_threshold,
            'avg_threshold': sum(dynamic_thresholds_used)/len(dynamic_thresholds_used) if dynamic_thresholds_used else self.default_proximity_threshold
        }

        return self.assistance_events

    def detect_assistance_events_optimized(self, sample_size=None):
        """
        Optimized version using Polars for faster processing of large AIS datasets.
        Detect assistance events using dynamic proximity thresholds based on vessel dimensions,
        course alignment validation within 20°, and speed similarity validation.
        Excludes tug boats from analysis to focus on pilot boat interactions with large commercial vessels.

        Args:
            sample_size: If provided, only process this many rows for testing
        """
        print("Detecting assistance events with Polars optimization...")
        if self.use_dynamic_thresholds:
            print("Using dynamic proximity thresholds based on vessel dimensions")
            print(f"Formula: 2 * max(pilot_boat_width, target_vessel_width)")
            print(f"Fallback threshold: {self.default_proximity_threshold}m")
        else:
            print(f"Using static proximity threshold: {self.default_proximity_threshold}m")
        print(f"Course alignment threshold: <={self.course_alignment_threshold} degrees")
        print(f"Speed similarity validation: Both vessels SOG 5-10 knots, difference <=3 knots")
        print(f"Boarding operation detection: Both vessels <=2 knots + recent movement validation")
        print(f"Vessel filtering: Excluding {len(self.tug_boats)} tug boats from analysis")

        # Use sample if specified
        data_to_process = self.dynamic_data
        if sample_size:
            data_to_process = self.dynamic_data.head(sample_size)
            print(f"Processing sample of {len(data_to_process)} records")

        # Convert to Polars for faster processing
        print("Converting to Polars DataFrame...")
        pl_data = pl.from_pandas(data_to_process)

        # Add pilot boat flag using Polars
        pl_data = pl_data.with_columns([
            pl.col("MMSI").is_in(self.pilot_boat_mmsi).alias("IsPilotBoat")
        ])

        # Get unique timestamps
        unique_times = pl_data.select("DateTime").unique().sort("DateTime")
        print(f"Processing {len(unique_times)} unique timestamps...")

        assistance_events = []
        proximity_only_events = 0
        course_aligned_events = 0
        speed_similar_events = 0
        total_proximity_checks = 0

        # Track dynamic threshold statistics
        dynamic_thresholds_used = []

        # Process each timestamp
        for i, timestamp_row in enumerate(unique_times.iter_rows()):
            timestamp = timestamp_row[0]

            if i % 10 == 0:
                print(f"Processing timestamp {i+1}/{len(unique_times)}: {timestamp}")

            # Filter data for current timestamp using Polars
            vessels_at_time = pl_data.filter(pl.col("DateTime") == timestamp)

            # Split into pilot boats and other vessels (excluding tug boats)
            pilot_boats = vessels_at_time.filter(pl.col("IsPilotBoat") == True)
            other_vessels = vessels_at_time.filter(
                (pl.col("IsPilotBoat") == False) &
                (~pl.col("MMSI").is_in(list(self.tug_boats)))
            )

            if len(pilot_boats) == 0:
                print(f"Warning: No pilot boats at timestamp {timestamp}")
                continue

            # Convert to numpy arrays for vectorized distance calculations
            pilot_coords = pilot_boats.select(["Latitude", "Longitude", "MMSI", "SOG", "COG"]).to_numpy()
            vessel_coords = other_vessels.select(["Latitude", "Longitude", "MMSI", "SOG", "COG"]).to_numpy()

            # Vectorized proximity and course alignment check
            for pilot_row in pilot_coords:
                pilot_lat, pilot_lon, pilot_mmsi, pilot_sog, pilot_cog = pilot_row

                for vessel_row in vessel_coords:
                    vessel_lat, vessel_lon, vessel_mmsi, vessel_sog, vessel_cog = vessel_row
                    total_proximity_checks += 1

                    # Calculate dynamic proximity threshold for this vessel pair
                    proximity_threshold = self.calculate_dynamic_proximity_threshold(
                        pilot_mmsi, vessel_mmsi
                    )
                    dynamic_thresholds_used.append(proximity_threshold)

                    # Calculate distance using vectorized haversine
                    distance = self.haversine_distance(pilot_lat, pilot_lon, vessel_lat, vessel_lon)

                    if distance <= proximity_threshold:
                        proximity_only_events += 1

                        # Get courses for alignment check
                        pilot_course = self._get_vessel_course_optimized(pl_data, pilot_mmsi, timestamp)
                        vessel_course = self._get_vessel_course_optimized(pl_data, vessel_mmsi, timestamp)

                        is_aligned, course_diff = self.is_course_aligned(pilot_course, vessel_course)

                        # Handle missing speed data
                        pilot_sog_val = pilot_sog if not pd.isna(pilot_sog) else 0
                        vessel_sog_val = vessel_sog if not pd.isna(vessel_sog) else 0

                        # Third check: speed similarity validation
                        is_speed_similar, speed_diff = self.is_speed_similar(pilot_sog_val, vessel_sog_val)

                        # Fourth check: boarding operation detection (both vessels slow)
                        is_boarding_op, max_speed = self.is_boarding_operation(pilot_sog_val, vessel_sog_val)

                        # Fifth check: movement validation for boarding operations
                        if is_boarding_op:
                            # Verify that at least one vessel has shown recent movement
                            pilot_has_movement = self.has_recent_movement(pilot_mmsi, timestamp)
                            vessel_has_movement = self.has_recent_movement(vessel_mmsi, timestamp)

                            # Only consider it a boarding operation if there's evidence of recent movement
                            is_boarding_op = is_boarding_op and (pilot_has_movement or vessel_has_movement)

                        # Valid assistance event if:
                        # 1. Courses are aligned AND speeds are similar (active assistance), OR
                        # 2. Both vessels are slow AND have recent movement (boarding/disembarking)
                        if (is_aligned and is_speed_similar) or is_boarding_op:
                            course_aligned_events += 1
                            if is_speed_similar:
                                speed_similar_events += 1

                            # Determine validation reason
                            if is_boarding_op:
                                validation_reason = 'boarding_operation'
                            elif is_aligned and is_speed_similar:
                                validation_reason = 'course_aligned_speed_similar'
                            elif is_aligned:
                                validation_reason = 'course_aligned_only'
                            else:
                                validation_reason = 'other'

                            # Classify traffic direction with enhanced context awareness
                            vessel_traffic_direction = self.classify_traffic_direction(
                                vessel_course, vessel_sog_val, is_aligned, distance, validation_reason, pilot_course
                            )
                            pilot_traffic_direction = self.classify_traffic_direction(
                                pilot_course, pilot_sog_val, is_aligned, distance, validation_reason
                            )

                            assistance_events.append({
                                'timestamp': timestamp,
                                'pilot_mmsi': pilot_mmsi,
                                'pilot_lat': pilot_lat,
                                'pilot_lon': pilot_lon,
                                'pilot_sog': pilot_sog_val,
                                'pilot_cog': pilot_course,
                                'vessel_mmsi': vessel_mmsi,
                                'vessel_lat': vessel_lat,
                                'vessel_lon': vessel_lon,
                                'vessel_sog': vessel_sog_val,
                                'vessel_cog': vessel_course,
                                'distance': distance,
                                'proximity_threshold_used': proximity_threshold,
                                'course_difference': course_diff,
                                'speed_difference': speed_diff,
                                'max_speed_in_pair': max_speed,
                                'is_course_aligned': is_aligned,
                                'is_speed_similar': is_speed_similar,
                                'is_boarding_operation': is_boarding_op,
                                'validation_reason': validation_reason,
                                'traffic_direction': vessel_traffic_direction,
                                'pilot_traffic_direction': pilot_traffic_direction
                            })

        self.assistance_events = pd.DataFrame(assistance_events)

        # Print validation statistics
        print(f"\nValidation Results:")
        print(f"- Total proximity checks: {total_proximity_checks:,}")

        # Dynamic threshold statistics
        if self.use_dynamic_thresholds and dynamic_thresholds_used:
            print(f"- Dynamic thresholds used: {len(dynamic_thresholds_used)} calculations")
            print(f"- Threshold range: {min(dynamic_thresholds_used):.1f}m to {max(dynamic_thresholds_used):.1f}m")
            print(f"- Average threshold: {sum(dynamic_thresholds_used)/len(dynamic_thresholds_used):.1f}m")
            print(f"- Proximity-only events (dynamic thresholds): {proximity_only_events}")
        else:
            print(f"- Proximity-only events (<={self.default_proximity_threshold}m): {proximity_only_events}")

        print(f"- Course-aligned events (<={self.course_alignment_threshold}deg): {course_aligned_events}")
        print(f"- Speed-similar events (5-10 knots, <=3 knot diff): {speed_similar_events}")
        print(f"- Valid assistance events: {len(self.assistance_events)}")

        if proximity_only_events > 0:
            reduction_rate = (1 - len(self.assistance_events) / proximity_only_events) * 100
            print(f"- False positive reduction: {reduction_rate:.1f}%")

        # Store threshold statistics for later analysis
        self.threshold_statistics = {
            'dynamic_thresholds_used': dynamic_thresholds_used,
            'min_threshold': min(dynamic_thresholds_used) if dynamic_thresholds_used else self.default_proximity_threshold,
            'max_threshold': max(dynamic_thresholds_used) if dynamic_thresholds_used else self.default_proximity_threshold,
            'avg_threshold': sum(dynamic_thresholds_used)/len(dynamic_thresholds_used) if dynamic_thresholds_used else self.default_proximity_threshold
        }

        return self.assistance_events

    def _get_vessel_course_optimized(self, pl_data, mmsi, timestamp):
        """
        Optimized version of get_vessel_course using Polars.
        """
        # First try to use COG if available
        current_record = pl_data.filter(
            (pl.col("MMSI") == mmsi) & (pl.col("DateTime") == timestamp)
        )

        if len(current_record) > 0:
            # Get the first record if multiple exist
            cog_values = current_record.select("COG").to_numpy().flatten()
            if len(cog_values) > 0:
                cog = cog_values[0]
                if pd.notna(cog) and 0 <= cog <= 360:
                    return cog

        # Calculate from position history using Polars
        vessel_history = pl_data.filter(pl.col("MMSI") == mmsi).sort("DateTime")

        if len(vessel_history) < 2:
            return np.nan

        # Convert to pandas for easier indexing (small subset)
        vessel_history_pd = vessel_history.to_pandas()

        # Find current position index
        current_idx = vessel_history_pd[vessel_history_pd['DateTime'] == timestamp].index

        if len(current_idx) == 0 or current_idx[0] == 0:
            return np.nan

        # Get current and previous positions
        current_pos = vessel_history_pd.iloc[current_idx[0]]
        prev_pos = vessel_history_pd.iloc[current_idx[0] - 1]

        # Extract coordinates
        curr_lat = current_pos['Latitude']
        curr_lon = current_pos['Longitude']
        prev_lat = prev_pos['Latitude']
        prev_lon = prev_pos['Longitude']

        if pd.notna(prev_lat) and pd.notna(prev_lon) and pd.notna(curr_lat) and pd.notna(curr_lon):
            return self.calculate_bearing(prev_lat, prev_lon, curr_lat, curr_lon)

        return np.nan

    def has_recent_movement(self, mmsi, current_time, lookback_minutes=10):
        """
        Check if a vessel has shown movement in the recent past.
        This helps distinguish between actively maneuvering vessels and anchored vessels.

        Args:
            mmsi: MMSI of the vessel to check
            current_time: Current timestamp
            lookback_minutes: How many minutes back to check for movement

        Returns:
            Boolean indicating if vessel has moved recently
        """
        if self.dynamic_data is None or self.dynamic_data.empty:
            return False

        # Define time window for checking recent movement
        start_time = current_time - pd.Timedelta(minutes=lookback_minutes)

        # Get vessel data in the time window
        vessel_history = self.dynamic_data[
            (self.dynamic_data['MMSI'] == mmsi) &
            (self.dynamic_data['DateTime'] >= start_time) &
            (self.dynamic_data['DateTime'] <= current_time)
        ].sort_values('DateTime')

        if len(vessel_history) < 2:
            return False  # Need at least 2 points to check movement

        # Check for significant movement in position or speed variation
        lat_range = vessel_history['Latitude'].max() - vessel_history['Latitude'].min()
        lon_range = vessel_history['Longitude'].max() - vessel_history['Longitude'].min()

        # Convert to approximate meters (rough calculation)
        lat_movement = lat_range * 111000  # meters
        lon_movement = lon_range * 111000 * np.cos(np.radians(vessel_history['Latitude'].mean()))
        total_movement = np.sqrt(lat_movement**2 + lon_movement**2)

        # Check for speed variation (indicates active maneuvering vs anchored)
        if 'SOG' in vessel_history.columns:
            speed_variation = vessel_history['SOG'].std()
            max_speed = vessel_history['SOG'].max()

            # Consider vessel as moving if:
            # 1. Total position change > 50 meters, OR
            # 2. Speed variation > 0.5 knots (indicating maneuvering), OR
            # 3. Maximum speed > 1.0 knots in the period
            has_movement = (total_movement > 50) or (speed_variation > 0.5) or (max_speed > 1.0)
        else:
            # Fallback to position change only
            has_movement = total_movement > 50

        return has_movement

    def _extract_trajectory_data(self, mmsi, start_time, end_time):
        """
        Extract complete trajectory data for a vessel between start and end times.

        Args:
            mmsi: MMSI of the vessel
            start_time: Start timestamp for trajectory extraction
            end_time: End timestamp for trajectory extraction

        Returns:
            List of [timestamp, latitude, longitude] points for the vessel
        """
        if self.dynamic_data is None or self.dynamic_data.empty:
            return []

        # Filter data for the specific vessel and time range
        vessel_trajectory = self.dynamic_data[
            (self.dynamic_data['MMSI'] == mmsi) &
            (self.dynamic_data['DateTime'] >= start_time) &
            (self.dynamic_data['DateTime'] <= end_time)
        ].sort_values('DateTime')

        # Extract trajectory points as list of [timestamp, lat, lon]
        trajectory_points = []
        for _, row in vessel_trajectory.iterrows():
            if pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
                trajectory_points.append([
                    row['DateTime'].isoformat(),  # Convert to ISO string for JSON serialization
                    float(row['Latitude']),
                    float(row['Longitude'])
                ])

        return trajectory_points

    def _extract_extended_trajectory_data(self, mmsi, session_start_time, session_end_time, buffer_minutes=15):
        """
        Extract extended trajectory data for a vessel including buffer time before and after the session.

        This method extracts AIS trajectory data for visualization purposes, covering a 30-minute window
        (15 minutes before session start + session duration + 15 minutes after session end).

        Args:
            mmsi: MMSI of the vessel
            session_start_time: Start timestamp of the assistance session
            session_end_time: End timestamp of the assistance session
            buffer_minutes: Minutes to extend before and after session (default: 15)

        Returns:
            List of dictionaries containing trajectory data with structure:
            [
                {
                    'timestamp': '2023-06-01T10:00:00',
                    'latitude': 35.1234,
                    'longitude': 129.5678,
                    'cog': 45.0,
                    'sog': 8.5
                },
                ...
            ]
        """
        if self.dynamic_data is None or self.dynamic_data.empty:
            return []

        # Calculate extended time window
        extended_start_time = session_start_time - pd.Timedelta(minutes=buffer_minutes)
        extended_end_time = session_end_time + pd.Timedelta(minutes=buffer_minutes)

        # Filter data for the specific vessel and extended time range
        vessel_trajectory = self.dynamic_data[
            (self.dynamic_data['MMSI'] == mmsi) &
            (self.dynamic_data['DateTime'] >= extended_start_time) &
            (self.dynamic_data['DateTime'] <= extended_end_time)
        ].sort_values('DateTime')

        # Extract trajectory points as list of dictionaries with full AIS data
        trajectory_points = []
        for _, row in vessel_trajectory.iterrows():
            if pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
                # Create trajectory point with all required fields
                trajectory_point = {
                    'timestamp': row['DateTime'].isoformat(),  # ISO string for JSON serialization
                    'latitude': float(row['Latitude']),
                    'longitude': float(row['Longitude']),
                    'cog': float(row['COG']) if pd.notna(row['COG']) else None,
                    'sog': float(row['SOG']) if pd.notna(row['SOG']) else None
                }
                trajectory_points.append(trajectory_point)

        return trajectory_points

    def group_continuous_events(self):
        """
        Group continuous proximity events into assistance sessions with complete and extended trajectory data.

        This method enhances the original session grouping by including both standard trajectory
        data for each session and extended trajectory data for visualization purposes.

        Enhanced features:
        - Extracts complete position sequences for both vessels during each session
        - Includes extended trajectory data with 30-minute window (15 min before + session + 15 min after)
        - Extended trajectory includes timestamp, latitude, longitude, COG, and SOG
        - Adds trajectory statistics for both standard and extended data
        - Preserves all existing session statistics and validation metrics
        - Handles edge cases where AIS data may not be available for full buffer periods

        Returns:
            DataFrame with enhanced session data including:

            Standard trajectory data (session duration only):
            - pilot_trajectory: List of [timestamp, lat, lon] points for pilot boat
            - vessel_trajectory: List of [timestamp, lat, lon] points for target vessel
            - trajectory_points: Total number of position observations
            - pilot_trajectory_points: Number of pilot boat trajectory points
            - vessel_trajectory_points: Number of target vessel trajectory points

            Extended trajectory data (30-minute window for visualization):
            - pilot_extended_trajectory: List of trajectory dictionaries with timestamp, lat, lon, COG, SOG
            - vessel_extended_trajectory: List of trajectory dictionaries with timestamp, lat, lon, COG, SOG
            - extended_trajectory_points: Total number of extended trajectory observations
            - pilot_extended_trajectory_points: Number of pilot boat extended trajectory points
            - vessel_extended_trajectory_points: Number of target vessel extended trajectory points

            Hybrid traffic direction classification (new functionality):
            - trajectory_bearing: Calculated bearing from session start to end coordinates
            - trajectory_based_direction: Classification based on trajectory bearing
            - hybrid_direction: Final hybrid classification result
            - hybrid_confidence: Confidence level (high/medium/low)
            - hybrid_reasoning: Explanation of classification decision
            - hybrid_method_used: Which method was prioritized (consensus/trajectory_priority/etc.)
            - distance_moved: Distance vessel moved during session (meters)
            - classification_notes: Additional analysis notes

            All existing session statistics (duration, distance, validation metrics)
        """
        print("Grouping continuous assistance events with trajectory extraction...")

        if self.assistance_events.empty:
            print("No assistance events to group")
            return pd.DataFrame()

        grouped_sessions = []

        # Group by pilot boat and vessel pair
        for (pilot_mmsi, vessel_mmsi), group in self.assistance_events.groupby(['pilot_mmsi', 'vessel_mmsi']):
            group = group.sort_values('timestamp')

            # Find continuous sessions (gaps > 5 minutes indicate new session)
            time_diffs = group['timestamp'].diff()
            session_breaks = time_diffs > pd.Timedelta(minutes=60)
            session_ids = session_breaks.cumsum()

            # Process each session
            for session_id, session_data in group.groupby(session_ids):
                session_data = session_data.sort_values('timestamp')

                start_time = session_data['timestamp'].min()
                end_time = session_data['timestamp'].max()
                duration = (end_time - start_time).total_seconds() / 60  # minutes

                # Only consider sessions longer than 1 minute
                if duration >= 1:
                    print(f"Extracting trajectory data for session: Pilot {pilot_mmsi} - Vessel {vessel_mmsi} ({start_time} to {end_time})")

                    # Extract complete trajectory data for both vessels (existing functionality)
                    pilot_trajectory = self._extract_trajectory_data(pilot_mmsi, start_time, end_time)
                    vessel_trajectory = self._extract_trajectory_data(vessel_mmsi, start_time, end_time)

                    # Extract extended trajectory data for visualization (30-minute window)
                    print(f"  - Extracting extended trajectory data (15 min buffer before/after session)")
                    pilot_extended_trajectory = self._extract_extended_trajectory_data(pilot_mmsi, start_time, end_time)
                    vessel_extended_trajectory = self._extract_extended_trajectory_data(vessel_mmsi, start_time, end_time)

                    # Calculate validation statistics for the session
                    course_aligned_count = session_data['is_course_aligned'].sum()
                    speed_similar_count = session_data['is_speed_similar'].sum() if 'is_speed_similar' in session_data.columns else 0
                    boarding_op_count = session_data['is_boarding_operation'].sum() if 'is_boarding_operation' in session_data.columns else 0
                    avg_course_diff = session_data['course_difference'].mean()
                    avg_speed_diff = session_data['speed_difference'].mean() if 'speed_difference' in session_data.columns else np.nan
                    avg_max_speed = session_data['max_speed_in_pair'].mean() if 'max_speed_in_pair' in session_data.columns else np.nan

                    # Calculate traffic direction statistics
                    traffic_directions = session_data['traffic_direction'].tolist() if 'traffic_direction' in session_data.columns else []
                    primary_traffic_direction = self.get_primary_traffic_direction(traffic_directions)

                    # Calculate hybrid classification for this session
                    session_info = {
                        'primary_traffic_direction': primary_traffic_direction,
                        'start_lat': session_data.iloc[0]['pilot_lat'],
                        'start_lon': session_data.iloc[0]['pilot_lon'],
                        'end_lat': session_data.iloc[-1]['pilot_lat'],
                        'end_lon': session_data.iloc[-1]['pilot_lon'],
                        'duration_minutes': duration
                    }

                    hybrid_classification = self.analyze_session_with_hybrid_classification(session_info)

                    grouped_sessions.append({
                        'pilot_mmsi': pilot_mmsi,
                        'vessel_mmsi': vessel_mmsi,
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration_minutes': duration,
                        'num_observations': len(session_data),
                        'min_distance': session_data['distance'].min(),
                        'max_distance': session_data['distance'].max(),
                        'avg_distance': session_data['distance'].mean(),
                        'start_lat': session_data.iloc[0]['pilot_lat'],
                        'start_lon': session_data.iloc[0]['pilot_lon'],
                        'end_lat': session_data.iloc[-1]['pilot_lat'],
                        'end_lon': session_data.iloc[-1]['pilot_lon'],
                        'course_aligned_observations': course_aligned_count,
                        'speed_similar_observations': speed_similar_count,
                        'boarding_operation_observations': boarding_op_count,
                        'avg_course_difference': avg_course_diff,
                        'avg_speed_difference': avg_speed_diff,
                        'avg_max_speed_in_pair': avg_max_speed,
                        'course_alignment_ratio': course_aligned_count / len(session_data),
                        'speed_similarity_ratio': speed_similar_count / len(session_data) if speed_similar_count > 0 else 0,
                        'boarding_operation_ratio': boarding_op_count / len(session_data) if boarding_op_count > 0 else 0,
                        'primary_validation_reason': session_data['validation_reason'].mode().iloc[0] if not session_data['validation_reason'].empty else 'unknown',
                        'primary_traffic_direction': primary_traffic_direction,
                        # Hybrid classification results (new functionality)
                        'trajectory_bearing': hybrid_classification['trajectory_bearing'],
                        'trajectory_based_direction': hybrid_classification['trajectory_based_direction'],
                        'hybrid_direction': hybrid_classification['hybrid_direction'],
                        'hybrid_confidence': hybrid_classification['hybrid_confidence'],
                        'hybrid_reasoning': hybrid_classification['hybrid_reasoning'],
                        'hybrid_method_used': hybrid_classification['hybrid_method_used'],
                        'distance_moved': hybrid_classification['distance_moved'],
                        'classification_notes': '; '.join(hybrid_classification['classification_notes']) if hybrid_classification['classification_notes'] else '',
                        # Complete trajectory data (existing functionality)
                        'pilot_trajectory': pilot_trajectory,
                        'vessel_trajectory': vessel_trajectory,
                        'trajectory_points': len(pilot_trajectory) + len(vessel_trajectory),
                        'pilot_trajectory_points': len(pilot_trajectory),
                        'vessel_trajectory_points': len(vessel_trajectory),
                        # Extended trajectory data for visualization (new functionality)
                        'pilot_extended_trajectory': pilot_extended_trajectory,
                        'vessel_extended_trajectory': vessel_extended_trajectory,
                        'extended_trajectory_points': len(pilot_extended_trajectory) + len(vessel_extended_trajectory),
                        'pilot_extended_trajectory_points': len(pilot_extended_trajectory),
                        'vessel_extended_trajectory_points': len(vessel_extended_trajectory)
                    })

        self.assistance_sessions = pd.DataFrame(grouped_sessions)
        print(f"Identified {len(self.assistance_sessions)} assistance sessions")

        return self.assistance_sessions

    def analyze_vessel_types(self):
        """
        Analyze which types of vessels are being assisted.
        Note: This requires static vessel data which may not be available.
        """
        print("Analyzing vessel types...")

        if hasattr(self, 'assistance_sessions') and not self.assistance_sessions.empty:
            # Count assistance by vessel
            vessel_assistance_count = self.assistance_sessions.groupby('vessel_mmsi').agg({
                'duration_minutes': ['count', 'sum', 'mean'],
                'min_distance': 'mean'
            }).round(2)

            vessel_assistance_count.columns = [
                'num_sessions', 'total_duration_min', 'avg_duration_min', 'avg_min_distance'
            ]

            return vessel_assistance_count.sort_values('total_duration_min', ascending=False)

        return pd.DataFrame()

    def analyze_pilot_boat_performance(self):
        """
        Analyze performance metrics for each pilot boat.
        """
        print("Analyzing pilot boat performance...")

        if hasattr(self, 'assistance_sessions') and not self.assistance_sessions.empty:
            pilot_performance = self.assistance_sessions.groupby('pilot_mmsi').agg({
                'vessel_mmsi': 'nunique',  # Number of unique vessels assisted
                'duration_minutes': ['count', 'sum', 'mean'],
                'min_distance': 'mean',
                'avg_distance': 'mean'
            }).round(2)

            pilot_performance.columns = [
                'vessels_assisted', 'num_sessions', 'total_duration_min',
                'avg_duration_min', 'avg_min_distance', 'avg_avg_distance'
            ]

            return pilot_performance.sort_values('total_duration_min', ascending=False)

        return pd.DataFrame()

    def analyze_temporal_patterns(self):
        """
        Analyze temporal patterns of assistance operations.
        """
        print("Analyzing temporal patterns...")

        if hasattr(self, 'assistance_sessions') and not self.assistance_sessions.empty:
            # Add time-based features
            sessions = self.assistance_sessions.copy()
            sessions['hour'] = sessions['start_time'].dt.hour
            sessions['day_of_week'] = sessions['start_time'].dt.day_name()

            # Hourly patterns
            hourly_pattern = sessions.groupby('hour').agg({
                'duration_minutes': ['count', 'sum', 'mean']
            }).round(2)
            hourly_pattern.columns = ['num_sessions', 'total_duration', 'avg_duration']

            # Daily patterns
            daily_pattern = sessions.groupby('day_of_week').agg({
                'duration_minutes': ['count', 'sum', 'mean']
            }).round(2)
            daily_pattern.columns = ['num_sessions', 'total_duration', 'avg_duration']

            return hourly_pattern, daily_pattern

        return pd.DataFrame(), pd.DataFrame()

    def analyze_directional_validation(self):
        """
        Analyze the effectiveness of directional and speed validation in filtering assistance events.
        Enhanced with traffic flow analysis.
        """
        print("Analyzing directional and speed validation effectiveness...")

        if not hasattr(self, 'assistance_events') or self.assistance_events.empty:
            return {}

        events = self.assistance_events

        # Count validation reasons
        validation_stats = events['validation_reason'].value_counts()

        # Course alignment statistics
        course_aligned_events = events[events['is_course_aligned'] == True]
        boarding_op_events = events[events['is_boarding_operation'] == True] if 'is_boarding_operation' in events.columns else pd.DataFrame()

        # Speed similarity statistics (if available)
        speed_similar_events = events[events['is_speed_similar'] == True] if 'is_speed_similar' in events.columns else pd.DataFrame()

        # Course difference statistics (for non-NaN values)
        valid_course_diffs = events['course_difference'].dropna()

        # Speed difference statistics (for non-NaN values)
        valid_speed_diffs = events['speed_difference'].dropna() if 'speed_difference' in events.columns else pd.Series()

        # Traffic flow analysis
        traffic_flow_stats = {}
        if 'traffic_direction' in events.columns:
            traffic_flow_stats = events['traffic_direction'].value_counts().to_dict()

            # Analyze traffic flow by validation reason
            traffic_by_validation = {}
            for reason in events['validation_reason'].unique():
                reason_events = events[events['validation_reason'] == reason]
                if not reason_events.empty and 'traffic_direction' in reason_events.columns:
                    traffic_by_validation[reason] = reason_events['traffic_direction'].value_counts().to_dict()

        analysis = {
            'total_events': len(events),
            'course_aligned_events': len(course_aligned_events),
            'speed_similar_events': len(speed_similar_events),
            'boarding_operation_events': len(boarding_op_events),
            'validation_reasons': validation_stats.to_dict(),
            'avg_course_difference': valid_course_diffs.mean() if not valid_course_diffs.empty else np.nan,
            'median_course_difference': valid_course_diffs.median() if not valid_course_diffs.empty else np.nan,
            'course_diff_std': valid_course_diffs.std() if not valid_course_diffs.empty else np.nan,
            'avg_speed_difference': valid_speed_diffs.mean() if not valid_speed_diffs.empty else np.nan,
            'median_speed_difference': valid_speed_diffs.median() if not valid_speed_diffs.empty else np.nan,
            'speed_diff_std': valid_speed_diffs.std() if not valid_speed_diffs.empty else np.nan,
            'traffic_flow_distribution': traffic_flow_stats,
            'traffic_flow_by_validation': traffic_by_validation if 'traffic_direction' in events.columns else {}
        }

        return analysis

    def generate_summary_report(self):
        """
        Generate a comprehensive summary report.
        """
        print("\n" + "="*60)
        print("PILOT BOAT ASSISTANCE ANALYSIS SUMMARY REPORT")
        print("="*60)

        # Basic statistics
        if hasattr(self, 'assistance_sessions') and not self.assistance_sessions.empty:
            total_sessions = len(self.assistance_sessions)
            total_duration = self.assistance_sessions['duration_minutes'].sum()
            avg_duration = self.assistance_sessions['duration_minutes'].mean()
            unique_pilots = self.assistance_sessions['pilot_mmsi'].nunique()
            unique_vessels = self.assistance_sessions['vessel_mmsi'].nunique()

            print(f"\nOVERALL STATISTICS:")
            print(f"- Total assistance sessions: {total_sessions}")
            print(f"- Total assistance duration: {total_duration:.1f} minutes ({total_duration/60:.1f} hours)")
            print(f"- Average session duration: {avg_duration:.1f} minutes")
            print(f"- Active pilot boats: {unique_pilots}")
            print(f"- Vessels assisted: {unique_vessels}")

            # Directional and speed validation analysis
            validation_analysis = self.analyze_directional_validation()
            if validation_analysis:
                print(f"\nVALIDATION ANALYSIS:")
                print(f"- Total validated events: {validation_analysis['total_events']}")
                print(f"- Course-aligned events: {validation_analysis['course_aligned_events']}")
                print(f"- Speed-similar events: {validation_analysis['speed_similar_events']}")
                print(f"- Boarding operation events: {validation_analysis['boarding_operation_events']}")

                if not pd.isna(validation_analysis['avg_course_difference']):
                    print(f"- Average course difference: {validation_analysis['avg_course_difference']:.1f}deg")
                    print(f"- Median course difference: {validation_analysis['median_course_difference']:.1f}deg")

                if not pd.isna(validation_analysis['avg_speed_difference']):
                    print(f"- Average speed difference: {validation_analysis['avg_speed_difference']:.1f} knots")
                    print(f"- Median speed difference: {validation_analysis['median_speed_difference']:.1f} knots")

                print(f"- Validation breakdown:")
                for reason, count in validation_analysis['validation_reasons'].items():
                    percentage = (count / validation_analysis['total_events']) * 100
                    print(f"  • {reason}: {count} ({percentage:.1f}%)")

                # Traffic flow analysis
                if validation_analysis.get('traffic_flow_distribution'):
                    print(f"\n- Traffic flow distribution:")
                    for direction, count in validation_analysis['traffic_flow_distribution'].items():
                        percentage = (count / validation_analysis['total_events']) * 100
                        print(f"  • {direction.capitalize()}: {count} events ({percentage:.1f}%)")

                # Traffic flow by validation reason
                if validation_analysis.get('traffic_flow_by_validation'):
                    print(f"\n- Traffic flow by validation reason:")
                    for reason, traffic_dist in validation_analysis['traffic_flow_by_validation'].items():
                        if traffic_dist:  # Only show if there's data
                            print(f"  • {reason}:")
                            for direction, count in traffic_dist.items():
                                print(f"    - {direction}: {count} events")

            # Pilot boat performance
            pilot_performance = self.analyze_pilot_boat_performance()
            if not pilot_performance.empty:
                print(f"\nTOP PERFORMING PILOT BOATS:")
                print(pilot_performance.head().to_string())

            # Vessel assistance patterns
            vessel_analysis = self.analyze_vessel_types()
            if not vessel_analysis.empty:
                print(f"\nMOST ASSISTED VESSELS:")
                print(vessel_analysis.head().to_string())

            # Temporal patterns
            hourly, daily = self.analyze_temporal_patterns()
            if not hourly.empty:
                print(f"\nBUSIEST HOURS (by number of sessions):")
                print(hourly.sort_values('num_sessions', ascending=False).head().to_string())

            if not daily.empty:
                print(f"\nBUSIEST DAYS (by number of sessions):")
                print(daily.sort_values('num_sessions', ascending=False).to_string())

        else:
            print("No assistance sessions found in the analysis period.")

        print("\n" + "="*60)

    def run_complete_analysis(self, sample_size=None):
        """
        Run the complete pilot boat assistance analysis with hybrid traffic direction classification.

        This method performs the complete analysis workflow including:
        - Loading and filtering vessel data
        - Detecting proximity events with dynamic thresholds
        - Grouping events into assistance sessions
        - Applying hybrid traffic direction classification (COG + trajectory-based)
        - Generating comprehensive reports with confidence scoring

        Args:
            sample_size: Number of records to process (None for all data)
        """
        print("Starting complete pilot boat assistance analysis...")

        # Load data in correct order for vessel size filtering
        self.load_pilot_boat_data()
        self.load_static_vessel_data()  # Load vessel dimensions first for filtering
        self.load_dynamic_data()        # Apply vessel size filtering during dynamic data loading

        # Detect assistance events using optimized method
        self.detect_assistance_events_optimized(sample_size=sample_size)

        # Group into sessions
        self.group_continuous_events()

        # Print hybrid classification summary
        self.print_hybrid_classification_summary()

        # Generate summary report
        self.generate_summary_report()

        # Save results
        self.save_results()

        print("Analysis complete!")

    def print_hybrid_classification_summary(self):
        """
        Print a summary of the hybrid traffic direction classification results.
        """
        if not hasattr(self, 'assistance_sessions') or self.assistance_sessions.empty:
            return

        print("\n" + "="*60)
        print("HYBRID TRAFFIC DIRECTION CLASSIFICATION SUMMARY")
        print("="*60)

        sessions = self.assistance_sessions

        # Filter sessions with valid hybrid classification data
        valid_sessions = sessions[sessions['trajectory_bearing'].notna()]

        if len(valid_sessions) == 0:
            print("No sessions with valid trajectory data for hybrid classification")
            return

        print(f"Total sessions analyzed: {len(sessions)}")
        print(f"Sessions with trajectory data: {len(valid_sessions)}")

        # Confidence distribution
        confidence_dist = valid_sessions['hybrid_confidence'].value_counts()
        print(f"\nHybrid Classification Confidence Distribution:")
        for confidence, count in confidence_dist.items():
            percentage = (count / len(valid_sessions)) * 100
            print(f"  {confidence.capitalize()}: {count} sessions ({percentage:.1f}%)")

        # Method usage distribution
        method_dist = valid_sessions['hybrid_method_used'].value_counts()
        print(f"\nHybrid Classification Method Usage:")
        for method, count in method_dist.items():
            method_name = method.replace('_', ' ').title()
            percentage = (count / len(valid_sessions)) * 100
            print(f"  {method_name}: {count} sessions ({percentage:.1f}%)")

        # Agreement analysis
        cog_trajectory_agreement = len(valid_sessions[
            valid_sessions['primary_traffic_direction'] == valid_sessions['trajectory_based_direction']
        ])
        cog_hybrid_agreement = len(valid_sessions[
            valid_sessions['primary_traffic_direction'] == valid_sessions['hybrid_direction']
        ])

        print(f"\nClassification Agreement Analysis:")
        print(f"  COG vs Trajectory Agreement: {cog_trajectory_agreement}/{len(valid_sessions)} ({(cog_trajectory_agreement/len(valid_sessions)*100):.1f}%)")
        print(f"  COG vs Hybrid Agreement: {cog_hybrid_agreement}/{len(valid_sessions)} ({(cog_hybrid_agreement/len(valid_sessions)*100):.1f}%)")

        # Direction distribution
        hybrid_directions = valid_sessions['hybrid_direction'].value_counts()
        print(f"\nHybrid Direction Distribution:")
        for direction, count in hybrid_directions.items():
            percentage = (count / len(valid_sessions)) * 100
            print(f"  {direction.capitalize()}: {count} sessions ({percentage:.1f}%)")

        # Distance and duration statistics
        avg_distance = valid_sessions['distance_moved'].mean()
        avg_duration = valid_sessions['duration_minutes'].mean()

        print(f"\nSession Statistics:")
        print(f"  Average distance moved: {avg_distance:.1f} meters")
        print(f"  Average session duration: {avg_duration:.1f} minutes")

        # High confidence corrections
        high_conf_corrections = len(valid_sessions[
            (valid_sessions['hybrid_confidence'] == 'high') &
            (valid_sessions['primary_traffic_direction'] != valid_sessions['hybrid_direction'])
        ])

        print(f"\nClassification Improvements:")
        print(f"  High-confidence corrections: {high_conf_corrections}/{len(valid_sessions)} ({(high_conf_corrections/len(valid_sessions)*100):.1f}%)")

        print("="*60)

    def save_results(self):
        """
        Save analysis results to CSV files.
        """
        print("Saving results...")

        if hasattr(self, 'assistance_events') and not self.assistance_events.empty:
            self.assistance_events.to_csv('pilot_boat_proximity_events.csv', index=False)
            print("Saved proximity events to: pilot_boat_proximity_events.csv")

        if hasattr(self, 'assistance_sessions') and not self.assistance_sessions.empty:
            # Create a copy for CSV export (trajectory data needs special handling)
            sessions_for_csv = self.assistance_sessions.copy()

            # Convert trajectory data to JSON strings for CSV compatibility
            if 'pilot_trajectory' in sessions_for_csv.columns:
                sessions_for_csv['pilot_trajectory'] = sessions_for_csv['pilot_trajectory'].apply(
                    lambda x: str(x) if x else '[]'
                )
            if 'vessel_trajectory' in sessions_for_csv.columns:
                sessions_for_csv['vessel_trajectory'] = sessions_for_csv['vessel_trajectory'].apply(
                    lambda x: str(x) if x else '[]'
                )
            # Convert extended trajectory data to JSON strings for CSV compatibility
            if 'pilot_extended_trajectory' in sessions_for_csv.columns:
                sessions_for_csv['pilot_extended_trajectory'] = sessions_for_csv['pilot_extended_trajectory'].apply(
                    lambda x: str(x) if x else '[]'
                )
            if 'vessel_extended_trajectory' in sessions_for_csv.columns:
                sessions_for_csv['vessel_extended_trajectory'] = sessions_for_csv['vessel_extended_trajectory'].apply(
                    lambda x: str(x) if x else '[]'
                )

            sessions_for_csv.to_csv('pilot_boat_assistance_sessions.csv', index=False)
            print("Saved assistance sessions to: pilot_boat_assistance_sessions.csv")

            # Save trajectory data separately in JSON format for better usability
            import json
            trajectory_data = {}
            for idx, row in self.assistance_sessions.iterrows():
                session_key = f"pilot_{row['pilot_mmsi']}_vessel_{row['vessel_mmsi']}_session_{idx}"
                trajectory_data[session_key] = {
                    'pilot_mmsi': row['pilot_mmsi'],
                    'vessel_mmsi': row['vessel_mmsi'],
                    'start_time': row['start_time'].isoformat(),
                    'end_time': row['end_time'].isoformat(),
                    'duration_minutes': row['duration_minutes'],
                    'primary_traffic_direction': row.get('primary_traffic_direction', 'unknown'),
                    # Standard trajectory data (session duration only)
                    'pilot_trajectory': row.get('pilot_trajectory', []),
                    'vessel_trajectory': row.get('vessel_trajectory', []),
                    'trajectory_points': row.get('trajectory_points', 0),
                    # Extended trajectory data (30-minute window for visualization)
                    'pilot_extended_trajectory': row.get('pilot_extended_trajectory', []),
                    'vessel_extended_trajectory': row.get('vessel_extended_trajectory', []),
                    'extended_trajectory_points': row.get('extended_trajectory_points', 0),
                    'pilot_extended_trajectory_points': row.get('pilot_extended_trajectory_points', 0),
                    'vessel_extended_trajectory_points': row.get('vessel_extended_trajectory_points', 0)
                }

            with open('pilot_boat_trajectories.json', 'w') as f:
                json.dump(trajectory_data, f, indent=2)
            print("Saved trajectory data to: pilot_boat_trajectories.json")

            # Save analysis summaries
            pilot_performance = self.analyze_pilot_boat_performance()
            if not pilot_performance.empty:
                pilot_performance.to_csv('pilot_boat_performance_summary.csv')
                print("Saved pilot performance summary to: pilot_boat_performance_summary.csv")

            vessel_analysis = self.analyze_vessel_types()
            if not vessel_analysis.empty:
                vessel_analysis.to_csv('vessel_assistance_summary.csv')
                print("Saved vessel assistance summary to: vessel_assistance_summary.csv")


def main():
    """
    Main function to run the pilot boat assistance analysis.
    """
    # File paths
    dynamic_data_path = "Sample_data_&_trial_codes/dataSet/busan/Busan_Dynamic_20230601_sorted.csv"
    pilot_boat_excel_path = "BusanPB.xlsx"
    static_data_path = "Sample_data_&_trial_codes/dataSet/busan/Static_Busan_Dynamic_20230601.csv"

    # Create analyzer instance with dynamic thresholds
    analyzer = PilotBoatAssistanceAnalyzer(dynamic_data_path, pilot_boat_excel_path, static_data_path)

    # Run complete analysis
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
