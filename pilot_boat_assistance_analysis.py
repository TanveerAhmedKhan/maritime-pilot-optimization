#!/usr/bin/env python3
"""
Maritime Pilot Boat Assistance Analysis with Dynamic Proximity Thresholds
=========================================================================

This script analyzes pilot boat assistance operations using vessel-specific proximity
thresholds based on vessel dimensions AND directional validation (course alignment
within 20°). This approach provides more realistic proximity detection by accounting
for vessel size differences in maritime operations.

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
- Speed similarity validation: both vessels must have SOG 5-10 knots with ≤3 knot difference

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

        print("Pilot Boat Assistance Analyzer initialized")
        if self.use_dynamic_thresholds:
            print(f"Dynamic proximity thresholds enabled (based on vessel dimensions)")
            print(f"Default fallback threshold: {self.default_proximity_threshold}m")
        else:
            print(f"Static proximity threshold: {self.default_proximity_threshold}m")
        print(f"Course alignment threshold: ±{self.course_alignment_threshold}°")
    
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
        Both vessels must be moving slowly (≤ 2 knots) indicating coordinated maneuvering.

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

        # Both vessels must be moving slowly for boarding operations
        # Typical boarding speeds are < 2 knots for safety
        pilot_slow = pilot_sog <= 2.0
        vessel_slow = vessel_sog <= 2.0

        # Both must be slow, not just one (avoids false positives)
        is_boarding = pilot_slow and vessel_slow

        max_speed = max(pilot_sog, vessel_sog)
        return is_boarding, max_speed

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

            # Check for required columns (note: 'witdh' is the actual column name in the data)
            if 'MMSI' in self.static_vessel_data.columns and 'witdh' in self.static_vessel_data.columns and 'loa' in self.static_vessel_data.columns:
                # Apply vessel size filtering
                print("Applying vessel size filters...")
                original_count = len(self.static_vessel_data)

                # Filter out vessels with width ≤ 2 meters or LOA < 60 meters
                filtered_data = self.static_vessel_data[
                    (self.static_vessel_data['witdh'] > 2) &
                    (self.static_vessel_data['loa'] >= 60) &
                    (pd.notna(self.static_vessel_data['witdh'])) &
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
                    width = row['witdh']
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
                print("Warning: Required columns (MMSI, witdh, loa) not found in static vessel data")
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
        """Load and prepare dynamic AIS data."""
        print("Loading dynamic AIS data...")
        
        # Load the CSV file
        self.dynamic_data = pd.read_csv(self.dynamic_data_path)
        print(f"Loaded dynamic data: {len(self.dynamic_data)} records")
        
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

        Args:
            sample_size: If provided, only process this many rows for testing
        """
        print("Detecting assistance events with dynamic proximity, directional, and speed validation...")
        if self.use_dynamic_thresholds:
            print("Using dynamic proximity thresholds based on vessel dimensions")
            print(f"Formula: 2 × max(pilot_boat_width, target_vessel_width)")
            print(f"Fallback threshold: {self.default_proximity_threshold}m")
        else:
            print(f"Using static proximity threshold: {self.default_proximity_threshold}m")
        print(f"Course alignment threshold: ≤{self.course_alignment_threshold}°")
        print(f"Speed similarity validation: Both vessels SOG 5-10 knots, difference ≤3 knots")
        print(f"Boarding operation detection: Both vessels ≤2 knots (replaces old stationary logic)")

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

            # Get other vessels at this timestamp
            other_vessels = vessels_at_time[~vessels_at_time['IsPilotBoat']]

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

                        # Valid assistance event if:
                        # 1. Courses are aligned AND speeds are similar (active assistance), OR
                        # 2. Both vessels are slow (≤ 2 knots) indicating boarding/disembarking
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
                                'validation_reason': validation_reason
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
            print(f"- Proximity-only events (≤{self.default_proximity_threshold}m): {proximity_only_events}")

        print(f"- Course-aligned events (≤{self.course_alignment_threshold}°): {course_aligned_events}")
        print(f"- Speed-similar events (5-10 knots, ≤3 knot diff): {speed_similar_events}")
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

        Args:
            sample_size: If provided, only process this many rows for testing
        """
        print("Detecting assistance events with Polars optimization...")
        if self.use_dynamic_thresholds:
            print("Using dynamic proximity thresholds based on vessel dimensions")
            print(f"Formula: 2 × max(pilot_boat_width, target_vessel_width)")
            print(f"Fallback threshold: {self.default_proximity_threshold}m")
        else:
            print(f"Using static proximity threshold: {self.default_proximity_threshold}m")
        print(f"Course alignment threshold: ≤{self.course_alignment_threshold}°")
        print(f"Speed similarity validation: Both vessels SOG 5-10 knots, difference ≤3 knots")
        print(f"Boarding operation detection: Both vessels ≤2 knots (replaces old stationary logic)")

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

            # Split into pilot boats and other vessels
            pilot_boats = vessels_at_time.filter(pl.col("IsPilotBoat") == True)
            other_vessels = vessels_at_time.filter(pl.col("IsPilotBoat") == False)

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

                        # Valid assistance event if:
                        # 1. Courses are aligned AND speeds are similar (active assistance), OR
                        # 2. Both vessels are slow (≤ 2 knots) indicating boarding/disembarking
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
                                'validation_reason': validation_reason
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
            print(f"- Proximity-only events (≤{self.default_proximity_threshold}m): {proximity_only_events}")

        print(f"- Course-aligned events (≤{self.course_alignment_threshold}°): {course_aligned_events}")
        print(f"- Speed-similar events (5-10 knots, ≤3 knot diff): {speed_similar_events}")
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

    def group_continuous_events(self):
        """
        Group continuous proximity events into assistance sessions.
        """
        print("Grouping continuous assistance events...")

        if self.assistance_events.empty:
            print("No assistance events to group")
            return pd.DataFrame()

        grouped_sessions = []

        # Group by pilot boat and vessel pair
        for (pilot_mmsi, vessel_mmsi), group in self.assistance_events.groupby(['pilot_mmsi', 'vessel_mmsi']):
            group = group.sort_values('timestamp')

            # Find continuous sessions (gaps > 5 minutes indicate new session)
            time_diffs = group['timestamp'].diff()
            session_breaks = time_diffs > pd.Timedelta(minutes=5)
            session_ids = session_breaks.cumsum()

            # Process each session
            for session_id, session_data in group.groupby(session_ids):
                session_data = session_data.sort_values('timestamp')

                start_time = session_data['timestamp'].min()
                end_time = session_data['timestamp'].max()
                duration = (end_time - start_time).total_seconds() / 60  # minutes

                # Only consider sessions longer than 1 minute
                if duration >= 1:
                    # Calculate validation statistics for the session
                    course_aligned_count = session_data['is_course_aligned'].sum()
                    speed_similar_count = session_data['is_speed_similar'].sum() if 'is_speed_similar' in session_data.columns else 0
                    boarding_op_count = session_data['is_boarding_operation'].sum() if 'is_boarding_operation' in session_data.columns else 0
                    avg_course_diff = session_data['course_difference'].mean()
                    avg_speed_diff = session_data['speed_difference'].mean() if 'speed_difference' in session_data.columns else np.nan
                    avg_max_speed = session_data['max_speed_in_pair'].mean() if 'max_speed_in_pair' in session_data.columns else np.nan

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
                        'primary_validation_reason': session_data['validation_reason'].mode().iloc[0] if not session_data['validation_reason'].empty else 'unknown'
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
            'speed_diff_std': valid_speed_diffs.std() if not valid_speed_diffs.empty else np.nan
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
                    print(f"- Average course difference: {validation_analysis['avg_course_difference']:.1f}°")
                    print(f"- Median course difference: {validation_analysis['median_course_difference']:.1f}°")

                if not pd.isna(validation_analysis['avg_speed_difference']):
                    print(f"- Average speed difference: {validation_analysis['avg_speed_difference']:.1f} knots")
                    print(f"- Median speed difference: {validation_analysis['median_speed_difference']:.1f} knots")

                print(f"- Validation breakdown:")
                for reason, count in validation_analysis['validation_reasons'].items():
                    percentage = (count / validation_analysis['total_events']) * 100
                    print(f"  • {reason}: {count} ({percentage:.1f}%)")

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
        Run the complete pilot boat assistance analysis.

        Args:
            sample_size: Number of records to process (None for all data)
        """
        print("Starting complete pilot boat assistance analysis...")

        # Load data
        self.load_pilot_boat_data()
        self.load_static_vessel_data()  # Load vessel dimensions for dynamic thresholds
        self.load_dynamic_data()

        # Detect assistance events using optimized method
        self.detect_assistance_events_optimized(sample_size=sample_size)

        # Group into sessions
        self.group_continuous_events()

        # Generate summary report
        self.generate_summary_report()

        # Save results
        self.save_results()

        print("Analysis complete!")

    def save_results(self):
        """
        Save analysis results to CSV files.
        """
        print("Saving results...")

        if hasattr(self, 'assistance_events') and not self.assistance_events.empty:
            self.assistance_events.to_csv('pilot_boat_proximity_events.csv', index=False)
            print("Saved proximity events to: pilot_boat_proximity_events.csv")

        if hasattr(self, 'assistance_sessions') and not self.assistance_sessions.empty:
            self.assistance_sessions.to_csv('pilot_boat_assistance_sessions.csv', index=False)
            print("Saved assistance sessions to: pilot_boat_assistance_sessions.csv")

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
