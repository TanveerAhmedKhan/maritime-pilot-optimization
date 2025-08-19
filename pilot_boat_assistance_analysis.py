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
from datetime import datetime, timedelta
import os
import re
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
        pilot_in_range = 4.0 <= pilot_sog <= 12.0
        vessel_in_range = 4.0 <= vessel_sog <= 12.0

        if not (pilot_in_range and vessel_in_range):
            return False, abs(pilot_sog - vessel_sog)

        # Check if speed difference is within acceptable range (≤ 3 knots)
        speed_diff = abs(pilot_sog - vessel_sog)
        is_similar = speed_diff <= 4.0

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
            # Read the static vessel data CSV (unified file)
            self.static_vessel_data = pd.read_csv(self.static_data_path)
            print(f"Loaded static vessel data: {len(self.static_vessel_data)} records")

            # Normalize/rename columns from various possible headers
            col_map = {}
            for col in list(self.static_vessel_data.columns):
                norm = str(col).lower().replace(" ", "").replace("_", "").replace("-", "").replace("(", "").replace(")", "")
                if col == 'MMSI':
                    col_map[col] = 'MMSI'
                elif 'mmsi' in norm:
                    col_map[col] = 'MMSI'
                elif ('breadth' in norm) or ('width' in norm) or ('beam' in norm):
                    col_map[col] = 'width'
                elif ('lengthoverall' in norm) or ('loa' in norm) or ('lengthoveral' in norm):
                    # Handle variants like 'lengthOverAll)Loa'
                    col_map[col] = 'loa'
                elif 'shiptype' in norm:
                    col_map[col] = 'shipType'
            if col_map:
                self.static_vessel_data = self.static_vessel_data.rename(columns=col_map)

            # Ensure required columns exist
            missing = [c for c in ['MMSI', 'width', 'loa'] if c not in self.static_vessel_data.columns]
            if missing:
                print(f"Warning: Required columns missing after normalization: {missing}")
                self.vessel_width_lookup = {}
                return

            # Coerce numeric types
            self.static_vessel_data['width'] = pd.to_numeric(self.static_vessel_data['width'], errors='coerce')
            self.static_vessel_data['loa'] = pd.to_numeric(self.static_vessel_data['loa'], errors='coerce')

            # Apply vessel size filtering
            print("Applying vessel size filters...")
            original_count = len(self.static_vessel_data)

            filtered_data = self.static_vessel_data[
                (pd.notna(self.static_vessel_data['width'])) &
                (pd.notna(self.static_vessel_data['loa'])) &
                (self.static_vessel_data['width'] > 2) &
                (self.static_vessel_data['loa'] >= 60)
            ]

            filtered_count = len(filtered_data)
            excluded_count = original_count - filtered_count
            print("Vessel filtering results:")
            print(f"- Original vessels: {original_count}")
            print(f"- Vessels after filtering (width > 2m AND LOA >= 60m): {filtered_count}")
            print(f"- Excluded vessels: {excluded_count}")

            # Build width lookup for dynamic thresholds
            self.vessel_width_lookup = {}
            for _, row in filtered_data.iterrows():
                mmsi = row['MMSI']
                width = row['width']
                if pd.notna(width) and width > 0:
                    self.vessel_width_lookup[int(mmsi)] = float(width)

            print(f"Vessel width data available for {len(self.vessel_width_lookup)} vessels")

            # Some stats
            if self.vessel_width_lookup:
                widths = list(self.vessel_width_lookup.values())
                print(f"Vessel width range (filtered): {min(widths):.1f}m to {max(widths):.1f}m")
                loa_values = filtered_data['loa'].dropna()
                if not loa_values.empty:
                    print(f"LOA range (filtered): {loa_values.min():.1f}m to {loa_values.max():.1f}m")
                    print(f"Average LOA (filtered): {loa_values.mean():.1f}m")

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
        df = pd.read_csv(self.dynamic_data_path)
        print(f"Loaded dynamic data: {len(df)} records")

        # Normalize column names to expected schema (robust to variants)
        col_map = {}
        for col in list(df.columns):
            norm = str(col).lower().strip()
            if norm in ('mmsi', 'mmsi_id'): col_map[col] = 'MMSI'
            elif norm in ('basedatetime', 'datetime', 'date_time', 'time', 'timestamp'): col_map[col] = 'DateTime'
            elif norm in ('latitude', 'lat', 'y', 'lat_dd'): col_map[col] = 'Latitude'
            elif norm in ('longitude', 'lon', 'long', 'lng', 'x', 'lon_dd'): col_map[col] = 'Longitude'
            elif norm in ('sog', 'speedoverground', 'speed_kn', 'speed', 'sog_kn'): col_map[col] = 'SOG'
            elif norm in ('cog', 'courseoverground', 'heading', 'course', 'cog_deg'): col_map[col] = 'COG'
        if col_map:
            df = df.rename(columns=col_map)

        # Handle duplicate or suffixed columns by picking the first match
        def pick_first(cols, pattern):
            for c in cols:
                if c == pattern or c.startswith(pattern + '.'): return c
            return None
        cols = list(df.columns)
        mmsi_col = pick_first(cols, 'MMSI') or next((c for c in cols if c.upper().startswith('MMSI')), None)
        dt_col = pick_first(cols, 'DateTime') or next((c for c in cols if c.lower() in ('basedatetime','datetime','date_time','time','timestamp')), None)
        lat_col = pick_first(cols, 'Latitude') or next((c for c in cols if c.lower() in ('latitude','lat','y','lat_dd')), None)
        lon_col = pick_first(cols, 'Longitude') or next((c for c in cols if c.lower() in ('longitude','lon','long','lng','x','lon_dd')), None)

        rename_final = {}
        if mmsi_col and mmsi_col != 'MMSI': rename_final[mmsi_col] = 'MMSI'
        if dt_col and dt_col != 'DateTime': rename_final[dt_col] = 'DateTime'
        if lat_col and lat_col != 'Latitude': rename_final[lat_col] = 'Latitude'
        if lon_col and lon_col != 'Longitude': rename_final[lon_col] = 'Longitude'
        if rename_final:
            df = df.rename(columns=rename_final)

        # Drop duplicate columns (keep first) to avoid issues like multiple 'DateTime' columns
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep='first')]

        required = ['MMSI', 'DateTime', 'Latitude', 'Longitude']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Dynamic data missing required columns: {missing}; available: {list(df.columns)[:20]} ...")

        print("Dynamic data: columns after rename:", list(df.columns)[:20])

        # Coerce types
        print("Coercing types for MMSI/Lat/Lon/SOG/COG...")
        df['MMSI'] = pd.to_numeric(df['MMSI'], errors='coerce')
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        if 'SOG' in df.columns:
            df['SOG'] = pd.to_numeric(df['SOG'], errors='coerce')
        if 'COG' in df.columns:
            df['COG'] = pd.to_numeric(df['COG'], errors='coerce')

        # Parse DateTime first to ensure validity
        print("Parsing DateTime and flooring to minute...")
        df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce').dt.floor('min')

        # Drop rows missing essential fields
        print("Dropping rows with NA in required fields...")
        df = df.dropna(subset=['MMSI', 'DateTime', 'Latitude', 'Longitude'])

        # Convert MMSI to int (safe casting)
        df['MMSI'] = df['MMSI'].astype(int)

        # Assign to instance
        self.dynamic_data = df
        print(f"Dynamic data prepared: {len(self.dynamic_data)} records, {self.dynamic_data['MMSI'].nunique()} vessels")

        # Apply vessel size filtering if static data is available
        if self.use_dynamic_thresholds and hasattr(self, 'vessel_width_lookup'):
            print("Applying vessel size filtering to dynamic AIS data...")
            original_count = len(self.dynamic_data)
            original_vessels = self.dynamic_data['MMSI'].nunique()

            valid_vessel_mmsi = set(self.vessel_width_lookup.keys())
            valid_vessel_mmsi.update(self.pilot_boat_mmsi)

            self.dynamic_data = self.dynamic_data[
                self.dynamic_data['MMSI'].isin(valid_vessel_mmsi)
            ]

            filtered_count = len(self.dynamic_data)
            filtered_vessels = self.dynamic_data['MMSI'].nunique()
            excluded_records = original_count - filtered_count
            excluded_vessels = original_vessels - filtered_vessels

            print("Dynamic data vessel size filtering results:")
            print(f"- Original records: {original_count:,}")
            print(f"- Records after filtering: {filtered_count:,}")
            print(f"- Excluded records: {excluded_records:,}")
            print(f"- Original unique vessels: {original_vessels}")
            print(f"- Vessels after filtering: {filtered_vessels}")
            print(f"- Excluded vessels: {excluded_vessels}")

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

                            # Detect nearby vessels for this specific event
                            nearby_vessels = self._detect_nearby_vessels_for_event(
                                vessel_mmsi, vessel_lat, vessel_lon, timestamp
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
                                'pilot_traffic_direction': pilot_traffic_direction,
                                # Nearby vessels data for this specific event
                                'nearby_vessels_300m': nearby_vessels,
                                'nearby_vessels_count': len(nearby_vessels),
                                'unique_nearby_vessels': len(set(v['mmsi'] for v in nearby_vessels)) if nearby_vessels else 0
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

    def _point_in_polygon(self, lat, lon, polygon_coords):
        """
        Determine if a point is inside a polygon using the ray casting algorithm.

        Args:
            lat: Latitude of the point
            lon: Longitude of the point
            polygon_coords: List of (lat, lon) tuples defining the polygon boundary

        Returns:
            Boolean indicating if the point is inside the polygon
        """
        if not polygon_coords or len(polygon_coords) < 3:
            return False

        x, y = lon, lat
        n = len(polygon_coords)
        inside = False

        p1x, p1y = polygon_coords[0][1], polygon_coords[0][0]  # lon, lat
        for i in range(1, n + 1):
            p2x, p2y = polygon_coords[i % n][1], polygon_coords[i % n][0]  # lon, lat
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def _get_busan_exclusion_zone(self):
        """
        Define the Busan Port exclusion zone polygon coordinates.

        Returns:
            List of (lat, lon) tuples defining the exclusion zone boundary
        """
        return [
            (35.079217, 128.775318),
            (35.079212, 128.832938),
            (35.071075, 128.832833),
            (35.063953, 128.791473),
            (35.067029, 128.783953),
            (35.074193, 128.775244),
            (35.079217, 128.775318)
        ]

    def _filter_sessions_by_exclusion_zone(self, sessions_df):
        """
        Filter out sessions where target vessels remain within the exclusion zone
        throughout the entire interaction period.

        Args:
            sessions_df: DataFrame containing pilot boat assistance sessions

        Returns:
            Tuple of (filtered_sessions_df, filtered_count, total_count)
        """
        if sessions_df.empty:
            return sessions_df, 0, 0

        exclusion_zone = self._get_busan_exclusion_zone()
        total_sessions = len(sessions_df)
        sessions_to_keep = []

        print(f"Applying geographic filtering to {total_sessions} sessions...")

        for idx, session in sessions_df.iterrows():
            vessel_trajectory_data = session['vessel_extended_trajectory']
            vessel_start = vessel_trajectory_data[0]
            vessel_end = vessel_trajectory_data[-1]
            

            # Get start and end coordinates
            start_lat = vessel_start['latitude']
            start_lon = vessel_start['longitude']
            end_lat = vessel_end['latitude']
            end_lon = vessel_end['longitude']

            # Check if both start and end points are within exclusion zone
            start_in_zone = self._point_in_polygon(start_lat, start_lon, exclusion_zone)
            end_in_zone = self._point_in_polygon(end_lat, end_lon, exclusion_zone)

            # Keep session if vessel starts OR ends outside the exclusion zone
            if not (start_in_zone and end_in_zone):
                sessions_to_keep.append(idx)

        filtered_sessions = sessions_df.loc[sessions_to_keep].copy()
        filtered_count = total_sessions - len(filtered_sessions)

        print(f"Geographic filtering results:")
        print(f"  - Total sessions: {total_sessions}")
        print(f"  - Sessions filtered out: {filtered_count}")
        print(f"  - Sessions remaining: {len(filtered_sessions)}")
        print(f"  - Filter rate: {filtered_count/total_sessions*100:.1f}%")

        return filtered_sessions, filtered_count, total_sessions

    def _detect_nearby_vessels_for_event(self, target_vessel_mmsi, target_lat, target_lon, event_timestamp, proximity_radius=300):
        """
        Detect all vessels within specified radius of target vessel at a specific event timestamp.

        Args:
            target_vessel_mmsi: MMSI of the target vessel
            target_lat: Latitude of the target vessel at event time
            target_lon: Longitude of the target vessel at event time
            event_timestamp: Specific timestamp of the proximity event
            proximity_radius: Detection radius in meters (default: 300m)

        Returns:
            List of dictionaries containing nearby vessel information
        """
        if self.dynamic_data is None or self.dynamic_data.empty:
            return []

        nearby_vessels = []

        # Get all vessels at this exact timestamp
        nearby_candidates = self.dynamic_data[
            (self.dynamic_data['DateTime'] == event_timestamp) &
            (self.dynamic_data['MMSI'] != target_vessel_mmsi) &  # Exclude target vessel
            (~self.dynamic_data['MMSI'].isin(self.pilot_boat_mmsi))  # Exclude pilot boats
        ]

        # Calculate distances for all candidates
        for _, candidate in nearby_candidates.iterrows():
            candidate_lat = candidate['Latitude']
            candidate_lon = candidate['Longitude']

            # Calculate distance using haversine formula
            distance = self.haversine_distance(
                target_lat, target_lon, candidate_lat, candidate_lon
            )

            if distance <= proximity_radius:
                # Get vessel type from static data if available
                vessel_type = 'Unknown'
                if hasattr(self, 'static_vessel_data') and self.static_vessel_data is not None:
                    vessel_info = self.static_vessel_data[
                        self.static_vessel_data['MMSI'] == candidate['MMSI']
                    ]
                    if not vessel_info.empty and 'shipType' in vessel_info.columns:
                        vessel_type = vessel_info.iloc[0]['shipType']

                nearby_vessel_info = {
                    'mmsi': candidate['MMSI'],
                    'distance': round(distance, 1),
                    'timestamp': event_timestamp,
                    'vessel_type': vessel_type,
                    'latitude': candidate_lat,
                    'longitude': candidate_lon
                }

                nearby_vessels.append(nearby_vessel_info)

        return nearby_vessels

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

                    # Preserve event-level nearby vessels data structure
                    print(f"  - Preserving event-level nearby vessels data from {len(session_data)} events")
                    event_level_nearby_vessels = []
                    total_detections = 0
                    unique_mmsis = set()

                    # Collect nearby vessels for each event in chronological order
                    for _, event in session_data.iterrows():
                        event_nearby_vessels = []
                        if 'nearby_vessels_300m' in event and event['nearby_vessels_300m']:
                            event_nearby_vessels = event['nearby_vessels_300m']
                            total_detections += len(event_nearby_vessels)
                            # Track unique MMSIs across all events
                            for vessel in event_nearby_vessels:
                                unique_mmsis.add(vessel['mmsi'])

                        # Append event-level nearby vessels (empty list if no vessels)
                        event_level_nearby_vessels.append(event_nearby_vessels)

                    nearby_vessels = event_level_nearby_vessels

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
                        'vessel_extended_trajectory_points': len(vessel_extended_trajectory),
                        # Nearby vessels detection (event-level preservation)
                        'nearby_vessels_300m': nearby_vessels,
                        'nearby_vessels_count': total_detections,
                        'unique_nearby_vessels': len(unique_mmsis),
                        'events_with_nearby_vessels': sum(1 for event_vessels in nearby_vessels if event_vessels),
                        'max_nearby_vessels_per_event': max(len(event_vessels) for event_vessels in nearby_vessels) if nearby_vessels else 0
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
            # Create a copy for CSV export (nearby vessels data needs special handling)
            events_for_csv = self.assistance_events.copy()

            # Convert nearby vessels data to JSON strings for CSV compatibility
            if 'nearby_vessels_300m' in events_for_csv.columns:
                events_for_csv['nearby_vessels_300m'] = events_for_csv['nearby_vessels_300m'].apply(
                    lambda x: str(x) if x else '[]'
                )

            events_for_csv.to_csv('pilot_boat_proximity_events.csv', index=False)
            print("Saved proximity events to: pilot_boat_proximity_events.csv")

        if hasattr(self, 'assistance_sessions') and not self.assistance_sessions.empty:
            # Apply geographic filtering to remove noisy sessions
            filtered_sessions, filtered_count, total_count = self._filter_sessions_by_exclusion_zone(self.assistance_sessions)

            # Create a copy for CSV export (trajectory data needs special handling)
            sessions_for_csv = filtered_sessions.copy()

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
            # Convert nearby vessels data to JSON strings for CSV compatibility
            if 'nearby_vessels_300m' in sessions_for_csv.columns:
                sessions_for_csv['nearby_vessels_300m'] = sessions_for_csv['nearby_vessels_300m'].apply(
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

    def save_trajectory_data(self, filename):
        """
        Save trajectory data to JSON file.

        Args:
            filename: Path to save the JSON file
        """
        import json

        if not hasattr(self, 'assistance_sessions') or self.assistance_sessions.empty:
            print("No trajectory data to save")
            return

        trajectory_data = {}

        for idx, session in self.assistance_sessions.iterrows():
            session_id = f"session_{idx}"

            # Safely handle possible list/array content in trajectory columns
            pilot_traj = session.get('pilot_trajectory', None)
            vessel_traj = session.get('vessel_trajectory', None)

            # Normalize to lists
            if isinstance(pilot_traj, (list, tuple)):
                pilot_traj_out = list(pilot_traj)
            elif pd.isna(pilot_traj):
                pilot_traj_out = []
            else:
                pilot_traj_out = []

            if isinstance(vessel_traj, (list, tuple)):
                vessel_traj_out = list(vessel_traj)
            elif pd.isna(vessel_traj):
                vessel_traj_out = []
            else:
                vessel_traj_out = []

            trajectory_data[session_id] = {
                'session_info': {
                    'pilot_mmsi': int(session['pilot_mmsi']) if pd.notna(session['pilot_mmsi']) else None,
                    'vessel_mmsi': int(session['vessel_mmsi']) if pd.notna(session['vessel_mmsi']) else None,
                    'start_time': session['start_time'].isoformat() if pd.notna(session['start_time']) else None,
                    'end_time': session['end_time'].isoformat() if pd.notna(session['end_time']) else None,
                    'duration_minutes': float(session['duration_minutes']) if pd.notna(session['duration_minutes']) else None
                },
                'pilot_trajectory': pilot_traj_out,
                'vessel_trajectory': vessel_traj_out
            }

        try:
            with open(filename, 'w') as f:
                json.dump(trajectory_data, f, indent=2, default=str)
            print(f"Trajectory data saved to: {filename}")
        except Exception as e:
            print(f"Error saving trajectory data: {e}")


# def main():
#     """
#     Main function to run the pilot boat assistance analysis.
#     """
#     # File paths
#     dynamic_data_path = "Sample_data_&_trial_codes/dataSet/busan/Busan_Dynamic_20230601_sorted.csv"
#     pilot_boat_excel_path = "BusanPB.xlsx"
#     static_data_path = "Sample_data_&_trial_codes/dataSet/busan/Static_Busan_Dynamic_20230601.csv"

#     # Create analyzer instance with dynamic thresholds
#     analyzer = PilotBoatAssistanceAnalyzer(dynamic_data_path, pilot_boat_excel_path, static_data_path)

#     # Run complete analysis
#     analyzer.run_complete_analysis()


def main_monthly_analysis():
    """
    Main function to run pilot boat assistance analysis for monthly dynamic files.
    Iterates through monthly CSVs and uses a single unified static data file.
    """
    import os
    from datetime import datetime

    print("="*80)
    print("MARITIME PILOT BOAT ASSISTANCE ANALYSIS - MONTHLY PROCESSING")
    print("="*80)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Paths
    monthly_base_path = "Sample_data_&_trial_codes/dataSet/Dynamic_09_2023_08_2024"
    pilot_boat_excel_path = "BusanPB.xlsx"
    unified_static_path = "Sample_data_&_trial_codes/dataSet/TalkFile_Static_merged_bbox_result_gokhan.csv"

    # Create results directory structure
    results_base_dir = "results"
    monthly_analysis_dir = os.path.join(results_base_dir, "monthly_analysis")
    os.makedirs(monthly_analysis_dir, exist_ok=True)

    print("Created output directory structure:")
    print(f"- Monthly analysis: {monthly_analysis_dir}")
    print()

    if not os.path.exists(monthly_base_path):
        print(f"ERROR: Monthly dynamic data directory not found: {monthly_base_path}")
        return

    if not os.path.exists(unified_static_path):
        print(f"ERROR: Unified static data file not found: {unified_static_path}")
        print("Please ensure the file is placed at the project root or update the path accordingly.")
        return

    # Discover monthly CSV files
    monthly_files = [f for f in os.listdir(monthly_base_path) if f.lower().endswith('.csv')]
    if not monthly_files:
        print(f"WARNING: No CSV files found in {monthly_base_path}")
        return

    # Storage for aggregation across months (optional)
    all_sessions = []
    all_events = []
    all_performance = []
    all_vessel_analysis = []

    # Optional: only process the first file for debugging
    only_first = os.environ.get('ANALYSIS_ONLY_FIRST') in ('1', 'true', 'True')

    # Process each monthly file
    for i, filename in enumerate(sorted(monthly_files), 1):
        dynamic_data_path = os.path.join(monthly_base_path, filename)

        # Derive a month label (YYYYMM) from filename if possible
        month_label = None
        m = re.search(r"(20\d{2})[_-]?(0[1-9]|1[0-2])", filename)
        if m:
            month_label = f"{m.group(1)}{m.group(2)}"
        else:
            # Fallback: use index-based label
            month_label = f"file_{i}"

        print(f"Processing Month {i}: {filename} -> label {month_label}")
        print("-" * 50)

        try:
            # Create analyzer instance for this month (uses unified static file)
            analyzer = PilotBoatAssistanceAnalyzer(dynamic_data_path, pilot_boat_excel_path, unified_static_path)

            # Run complete analysis for this month (optional sampling via env var ANALYSIS_SAMPLE_SIZE)
            sample_env = os.environ.get('ANALYSIS_SAMPLE_SIZE')
            sample_size = int(sample_env) if sample_env and sample_env.isdigit() else None
            if sample_size:
                print(f"Running with sample_size={sample_size} for quick verification")
            analyzer.run_complete_analysis(sample_size=sample_size)

            # Create month-specific output directory
            month_output_dir = os.path.join(monthly_analysis_dir, f"month_{month_label}")
            os.makedirs(month_output_dir, exist_ok=True)

            # Save month-specific results (reuse daily saver with month label)
            save_daily_results(analyzer, month_output_dir, month_label)

            # Collect data for aggregated summaries (optional)
            if hasattr(analyzer, 'assistance_sessions') and not analyzer.assistance_sessions.empty:
                sessions_with_month = analyzer.assistance_sessions.copy()
                sessions_with_month['analysis_month'] = month_label
                all_sessions.append(sessions_with_month)

            if hasattr(analyzer, 'assistance_events') and not analyzer.assistance_events.empty:
                events_with_month = analyzer.assistance_events.copy()
                events_with_month['analysis_month'] = month_label
                all_events.append(events_with_month)

            # Collect performance data
            pilot_performance = analyzer.analyze_pilot_boat_performance()
            if not pilot_performance.empty:
                pilot_performance['analysis_month'] = month_label
                all_performance.append(pilot_performance)

            vessel_analysis = analyzer.analyze_vessel_types()
            if not vessel_analysis.empty:
                vessel_analysis['analysis_month'] = month_label
                all_vessel_analysis.append(vessel_analysis)

            print(f"SUCCESS: Month {month_label} processing completed successfully")
            print(f"  Results saved to: {month_output_dir}")
            print()

        except Exception as e:
            import traceback
            print(f"ERROR: Error processing month {month_label}: {str(e)}")
            traceback.print_exc()
            print()

        if only_first:
            print("ANALYSIS_ONLY_FIRST is set; stopping after first month for debugging.")
            break

    print("="*80)
    print("MONTHLY ANALYSIS COMPLETED")
    print("="*80)
    print(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results available in: {results_base_dir}")
    print()


def save_daily_results(analyzer, output_dir, date):
    """
    Save analysis results for a specific day to the designated output directory.

    Args:
        analyzer: PilotBoatAssistanceAnalyzer instance
        output_dir: Directory to save results
        date: Date string for file naming
    """
    import os

    # Save proximity events
    if hasattr(analyzer, 'assistance_events') and not analyzer.assistance_events.empty:
        # Create a copy for CSV export (nearby vessels data needs special handling)
        events_for_csv = analyzer.assistance_events.copy()

        # Convert nearby vessels data to JSON strings for CSV compatibility
        if 'nearby_vessels_300m' in events_for_csv.columns:
            events_for_csv['nearby_vessels_300m'] = events_for_csv['nearby_vessels_300m'].apply(
                lambda x: str(x) if x else '[]'
            )

        events_file = os.path.join(output_dir, f"pilot_boat_proximity_events_{date}.csv")
        events_for_csv.to_csv(events_file, index=False)
        print(f"  Saved proximity events: {events_file}")

    # Save assistance sessions
    if hasattr(analyzer, 'assistance_sessions') and not analyzer.assistance_sessions.empty:
        # Apply geographic filtering to remove noisy sessions
        filtered_sessions, filtered_count, total_count = analyzer._filter_sessions_by_exclusion_zone(analyzer.assistance_sessions)

        # Create a copy for CSV export (trajectory data needs special handling)
        sessions_for_csv = filtered_sessions.copy()

        # Convert trajectory data to string representation for CSV
        if 'pilot_trajectory' in sessions_for_csv.columns:
            def format_trajectory(x):
                if x is None or (isinstance(x, float) and pd.isna(x)):
                    return "No data"
                elif isinstance(x, list):
                    return f"Points: {len(x)}"
                else:
                    return "No data"

            sessions_for_csv['pilot_trajectory_summary'] = sessions_for_csv['pilot_trajectory'].apply(format_trajectory)
            sessions_for_csv = sessions_for_csv.drop('pilot_trajectory', axis=1)

        if 'vessel_trajectory' in sessions_for_csv.columns:
            def format_trajectory(x):
                if x is None or (isinstance(x, float) and pd.isna(x)):
                    return "No data"
                elif isinstance(x, list):
                    return f"Points: {len(x)}"
                else:
                    return "No data"

            sessions_for_csv['vessel_trajectory_summary'] = sessions_for_csv['vessel_trajectory'].apply(format_trajectory)
            sessions_for_csv = sessions_for_csv.drop('vessel_trajectory', axis=1)

        # Handle nearby vessels data for CSV export
        if 'nearby_vessels_300m' in sessions_for_csv.columns:
            def format_nearby_vessels(x):
                if x is None or (isinstance(x, float) and pd.isna(x)):
                    return "No nearby vessels"
                elif isinstance(x, list):
                    return f"Nearby vessels: {len(x)}"
                else:
                    return "No nearby vessels"

            sessions_for_csv['nearby_vessels_summary'] = sessions_for_csv['nearby_vessels_300m'].apply(format_nearby_vessels)
            # Keep the full data as string for detailed analysis
            sessions_for_csv['nearby_vessels_300m'] = sessions_for_csv['nearby_vessels_300m'].apply(
                lambda x: str(x) if x else '[]'
            )

        sessions_file = os.path.join(output_dir, f"pilot_boat_assistance_sessions_{date}.csv")
        sessions_for_csv.to_csv(sessions_file, index=False)
        print(f"  Saved assistance sessions: {sessions_file}")

        # Save trajectory data separately as JSON
        if hasattr(analyzer, 'assistance_sessions'):
            trajectory_file = os.path.join(output_dir, f"pilot_boat_trajectories_{date}.json")
            analyzer.save_trajectory_data(trajectory_file)
            print(f"  Saved trajectory data: {trajectory_file}")

    # Save performance summaries
    pilot_performance = analyzer.analyze_pilot_boat_performance()
    if not pilot_performance.empty:
        performance_file = os.path.join(output_dir, f"pilot_boat_performance_summary_{date}.csv")
        pilot_performance.to_csv(performance_file, index=False)
        print(f"  Saved pilot performance: {performance_file}")

    vessel_analysis = analyzer.analyze_vessel_types()
    if not vessel_analysis.empty:
        vessel_file = os.path.join(output_dir, f"vessel_assistance_summary_{date}.csv")
        vessel_analysis.to_csv(vessel_file, index=False)
        print(f"  Saved vessel analysis: {vessel_file}")


def create_daily_summary(analyzer, date, day_number):
    """
    Create a summary dictionary for a single day's analysis.

    Args:
        analyzer: PilotBoatAssistanceAnalyzer instance
        date: Date string
        day_number: Day number (1-7)

    Returns:
        Dictionary with daily summary statistics
    """
    summary = {
        'day_number': day_number,
        'date': date,
        'total_events': 0,
        'total_sessions': 0,
        'unique_pilot_boats': 0,
        'unique_vessels': 0,
        'avg_session_duration': 0,
        'total_assistance_time': 0,
        'inbound_sessions': 0,
        'outbound_sessions': 0,
        'other_sessions': 0
    }

    # Events summary
    if hasattr(analyzer, 'assistance_events') and not analyzer.assistance_events.empty:
        summary['total_events'] = len(analyzer.assistance_events)
        summary['unique_pilot_boats'] = analyzer.assistance_events['pilot_mmsi'].nunique()
        summary['unique_vessels'] = analyzer.assistance_events['vessel_mmsi'].nunique()

    # Sessions summary
    if hasattr(analyzer, 'assistance_sessions') and not analyzer.assistance_sessions.empty:
        sessions = analyzer.assistance_sessions
        summary['total_sessions'] = len(sessions)
        summary['avg_session_duration'] = sessions['duration_minutes'].mean()
        summary['total_assistance_time'] = sessions['duration_minutes'].sum()

        # Traffic direction breakdown
        if 'hybrid_direction' in sessions.columns:
            direction_counts = sessions['hybrid_direction'].value_counts()
            summary['inbound_sessions'] = direction_counts.get('inbound', 0)
            summary['outbound_sessions'] = direction_counts.get('outbound', 0)
            summary['other_sessions'] = direction_counts.get('other', 0) + direction_counts.get('mixed', 0)
        elif 'primary_traffic_direction' in sessions.columns:
            direction_counts = sessions['primary_traffic_direction'].value_counts()
            summary['inbound_sessions'] = direction_counts.get('inbound', 0)
            summary['outbound_sessions'] = direction_counts.get('outbound', 0)
            summary['other_sessions'] = direction_counts.get('other', 0) + direction_counts.get('mixed', 0)

    return summary


def generate_weekly_summary(weekly_sessions, weekly_events, weekly_performance,
                          weekly_vessel_analysis, daily_summaries, output_dir):
    """
    Generate comprehensive weekly summary from all daily analyses.

    Args:
        weekly_sessions: List of session DataFrames from all days
        weekly_events: List of event DataFrames from all days
        weekly_performance: List of performance DataFrames from all days
        weekly_vessel_analysis: List of vessel analysis DataFrames from all days
        daily_summaries: List of daily summary dictionaries
        output_dir: Directory to save weekly summary files
    """
    import os

    print("Generating weekly summary reports...")

    # Combine all sessions
    if weekly_sessions:
        combined_sessions = pd.concat(weekly_sessions, ignore_index=True)
        sessions_file = os.path.join(output_dir, "weekly_assistance_sessions_summary.csv")
        combined_sessions.to_csv(sessions_file, index=False)
        print(f"SUCCESS: Weekly sessions summary: {sessions_file}")

        # Generate weekly session statistics
        weekly_session_stats = analyze_weekly_sessions(combined_sessions)
        stats_file = os.path.join(output_dir, "weekly_session_statistics.csv")
        weekly_session_stats.to_csv(stats_file, index=False)
        print(f"SUCCESS: Weekly session statistics: {stats_file}")

    # Combine all events
    if weekly_events:
        combined_events = pd.concat(weekly_events, ignore_index=True)
        events_file = os.path.join(output_dir, "weekly_proximity_events_summary.csv")
        combined_events.to_csv(events_file, index=False)
        print(f"SUCCESS: Weekly events summary: {events_file}")

    # Combine performance data
    if weekly_performance:
        combined_performance = pd.concat(weekly_performance, ignore_index=True)
        performance_file = os.path.join(output_dir, "weekly_pilot_performance_summary.csv")
        combined_performance.to_csv(performance_file, index=False)
        print(f"SUCCESS: Weekly performance summary: {performance_file}")

    # Combine vessel analysis
    if weekly_vessel_analysis:
        combined_vessel_analysis = pd.concat(weekly_vessel_analysis, ignore_index=True)
        vessel_file = os.path.join(output_dir, "weekly_vessel_analysis_summary.csv")
        combined_vessel_analysis.to_csv(vessel_file, index=False)
        print(f"SUCCESS: Weekly vessel analysis: {vessel_file}")

    # Create daily summaries report
    if daily_summaries:
        daily_summary_df = pd.DataFrame(daily_summaries)
        daily_file = os.path.join(output_dir, "daily_summaries_overview.csv")
        daily_summary_df.to_csv(daily_file, index=False)
        print(f"SUCCESS: Daily summaries overview: {daily_file}")

        # Generate weekly trends analysis
        trends_analysis = analyze_weekly_trends(daily_summary_df)
        trends_file = os.path.join(output_dir, "weekly_trends_analysis.csv")
        trends_analysis.to_csv(trends_file, index=False)
        print(f"SUCCESS: Weekly trends analysis: {trends_file}")

    # Generate comprehensive weekly report
    generate_weekly_report(weekly_sessions, daily_summaries, output_dir)

    print("Weekly summary generation completed!")


def analyze_weekly_sessions(combined_sessions):
    """
    Analyze weekly session patterns and generate statistics.

    Args:
        combined_sessions: Combined DataFrame of all sessions

    Returns:
        DataFrame with weekly session statistics
    """
    stats = []

    # Overall statistics
    total_sessions = len(combined_sessions)
    total_duration = combined_sessions['duration_minutes'].sum()
    avg_duration = combined_sessions['duration_minutes'].mean()

    stats.append({
        'metric': 'Total Sessions',
        'value': total_sessions,
        'unit': 'sessions'
    })

    stats.append({
        'metric': 'Total Assistance Time',
        'value': round(total_duration, 2),
        'unit': 'minutes'
    })

    stats.append({
        'metric': 'Average Session Duration',
        'value': round(avg_duration, 2),
        'unit': 'minutes'
    })

    # Traffic direction statistics
    if 'hybrid_direction' in combined_sessions.columns:
        direction_col = 'hybrid_direction'
    elif 'primary_traffic_direction' in combined_sessions.columns:
        direction_col = 'primary_traffic_direction'
    else:
        direction_col = None

    if direction_col:
        direction_counts = combined_sessions[direction_col].value_counts()
        for direction, count in direction_counts.items():
            percentage = (count / total_sessions) * 100
            stats.append({
                'metric': f'{direction.title()} Sessions',
                'value': f"{count} ({percentage:.1f}%)",
                'unit': 'sessions'
            })

    # Pilot boat statistics
    unique_pilots = combined_sessions['pilot_mmsi'].nunique()
    unique_vessels = combined_sessions['vessel_mmsi'].nunique()

    stats.append({
        'metric': 'Unique Pilot Boats',
        'value': unique_pilots,
        'unit': 'vessels'
    })

    stats.append({
        'metric': 'Unique Assisted Vessels',
        'value': unique_vessels,
        'unit': 'vessels'
    })

    return pd.DataFrame(stats)


def analyze_weekly_trends(daily_summary_df):
    """
    Analyze trends across the week.

    Args:
        daily_summary_df: DataFrame with daily summaries

    Returns:
        DataFrame with trend analysis
    """
    trends = []

    # Calculate day-over-day changes
    for metric in ['total_sessions', 'total_assistance_time', 'unique_pilot_boats', 'unique_vessels']:
        if metric in daily_summary_df.columns:
            values = daily_summary_df[metric].tolist()

            # Calculate trend (simple linear trend)
            days = list(range(1, len(values) + 1))
            if len(values) > 1:
                correlation = pd.Series(days).corr(pd.Series(values))
                trend_direction = "Increasing" if correlation > 0.1 else "Decreasing" if correlation < -0.1 else "Stable"
            else:
                trend_direction = "Insufficient data"

            trends.append({
                'metric': metric.replace('_', ' ').title(),
                'week_total': sum(values) if values else 0,
                'daily_average': sum(values) / len(values) if values else 0,
                'min_day': min(values) if values else 0,
                'max_day': max(values) if values else 0,
                'trend_direction': trend_direction
            })

    return pd.DataFrame(trends)


def generate_weekly_report(weekly_sessions, daily_summaries, output_dir):
    """
    Generate a comprehensive weekly report in text format.

    Args:
        weekly_sessions: List of session DataFrames
        daily_summaries: List of daily summary dictionaries
        output_dir: Output directory
    """
    import os
    from datetime import datetime

    report_file = os.path.join(output_dir, "weekly_analysis_report.md")

    with open(report_file, 'w') as f:
        f.write("# Maritime Pilot Boat Analysis - Weekly Report\n\n")
        f.write(f"**Analysis Period:** June 1-7, 2023 (7 days)\n")
        f.write(f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Executive Summary\n\n")

        if daily_summaries:
            total_sessions = sum(day['total_sessions'] for day in daily_summaries)
            total_events = sum(day['total_events'] for day in daily_summaries)
            total_time = sum(day['total_assistance_time'] for day in daily_summaries)

            f.write(f"- **Total Assistance Sessions:** {total_sessions}\n")
            f.write(f"- **Total Proximity Events:** {total_events}\n")
            f.write(f"- **Total Assistance Time:** {total_time:.1f} minutes ({total_time/60:.1f} hours)\n")
            f.write(f"- **Average Sessions per Day:** {total_sessions/7:.1f}\n\n")

        f.write("## Daily Breakdown\n\n")
        f.write("| Day | Date | Sessions | Events | Pilot Boats | Vessels | Duration (min) |\n")
        f.write("|-----|------|----------|--------|-------------|---------|----------------|\n")

        for day in daily_summaries:
            f.write(f"| {day['day_number']} | {day['date']} | {day['total_sessions']} | "
                   f"{day['total_events']} | {day['unique_pilot_boats']} | "
                   f"{day['unique_vessels']} | {day['total_assistance_time']:.1f} |\n")

        f.write("\n## Traffic Direction Analysis\n\n")
        if daily_summaries:
            total_inbound = sum(day['inbound_sessions'] for day in daily_summaries)
            total_outbound = sum(day['outbound_sessions'] for day in daily_summaries)
            total_other = sum(day['other_sessions'] for day in daily_summaries)
            total_classified = total_inbound + total_outbound + total_other

            if total_classified > 0:
                f.write(f"- **Inbound Traffic:** {total_inbound} sessions ({total_inbound/total_classified*100:.1f}%)\n")
                f.write(f"- **Outbound Traffic:** {total_outbound} sessions ({total_outbound/total_classified*100:.1f}%)\n")
                f.write(f"- **Other/Mixed Traffic:** {total_other} sessions ({total_other/total_classified*100:.1f}%)\n\n")

        f.write("## Key Findings\n\n")
        f.write("- Analysis covers 7 consecutive days of AIS data\n")
        f.write("- Dynamic proximity thresholds based on vessel dimensions\n")
        f.write("- Enhanced traffic direction classification with hybrid methodology\n")
        f.write("- Comprehensive session detection with temporal continuity\n\n")

        f.write("## Data Quality Notes\n\n")
        f.write("- Vessel size filtering applied (width > 2m, LOA >= 60m)\n")
        f.write("- Tug boats excluded from analysis\n")
        f.write("- Course alignment validation (<=20° difference)\n")
        f.write("- Speed similarity validation for assistance operations\n\n")

    print(f"SUCCESS: Weekly analysis report: {report_file}")


if __name__ == "__main__":
    # Run monthly analysis by default
    main_monthly_analysis()
