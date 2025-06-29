#!/usr/bin/env python3
"""
Generate fixed interactive web-based visualization for pilot boat assistance sessions.
"""

import pandas as pd
import json
import numpy as np
from datetime import datetime
import ast

def parse_trajectory_string(traj_str):
    """Parse trajectory string from CSV into list of coordinates."""
    if pd.isna(traj_str) or traj_str == '':
        return []
    try:
        # Parse the string representation of the list
        traj_list = ast.literal_eval(traj_str)
        # Convert to [lat, lon] format (removing timestamp if present)
        if traj_list and len(traj_list[0]) >= 3:
            # Format: [timestamp, lat, lon] -> [lat, lon]
            return [[point[1], point[2]] for point in traj_list if len(point) >= 3]
        elif traj_list and len(traj_list[0]) == 2:
            # Already in [lat, lon] format
            return traj_list
        return []
    except (ValueError, SyntaxError, IndexError):
        return []

def load_and_fix_data():
    """Load and fix data format issues."""
    print("Loading and fixing data files...")

    # Load sessions data
    sessions = pd.read_csv('pilot_boat_assistance_sessions.csv')
    print(f"Loaded {len(sessions)} sessions")

    # Load proximity events
    events = pd.read_csv('pilot_boat_proximity_events.csv')
    print(f"Loaded {len(events)} proximity events")

    # Load trajectory data
    with open('pilot_boat_trajectories.json', 'r') as f:
        trajectories = json.load(f)
    print(f"Loaded trajectory data for {len(trajectories)} sessions")

    # Parse trajectory strings in sessions data if they exist as strings
    if 'pilot_trajectory' in sessions.columns:
        sessions['pilot_trajectory_parsed'] = sessions['pilot_trajectory'].apply(parse_trajectory_string)
        sessions['vessel_trajectory_parsed'] = sessions['vessel_trajectory'].apply(parse_trajectory_string)
        print("Parsed trajectory strings from sessions data")
    
    # Fix data format issues
    print("Fixing data format issues...")
    
    # Fix MMSI data types
    sessions['pilot_mmsi'] = sessions['pilot_mmsi'].astype(int)
    sessions['vessel_mmsi'] = sessions['vessel_mmsi'].astype(int)
    events['pilot_mmsi'] = events['pilot_mmsi'].astype(int)
    events['vessel_mmsi'] = events['vessel_mmsi'].astype(int)
    
    # Fix trajectory data format
    fixed_trajectories = {}
    for key, traj_data in trajectories.items():
        fixed_traj = traj_data.copy()
        
        # Convert MMSI to int
        fixed_traj['pilot_mmsi'] = int(fixed_traj['pilot_mmsi'])
        fixed_traj['vessel_mmsi'] = int(fixed_traj['vessel_mmsi'])
        
        # Fix start_time format to match sessions
        if 'T' in fixed_traj['start_time']:
            fixed_traj['start_time'] = fixed_traj['start_time'].replace('T', ' ')
        if 'T' in fixed_traj['end_time']:
            fixed_traj['end_time'] = fixed_traj['end_time'].replace('T', ' ')
        
        # Fix trajectory coordinates - remove timestamp and keep only [lat, lon]
        if 'pilot_trajectory' in fixed_traj and fixed_traj['pilot_trajectory']:
            pilot_coords = []
            for point in fixed_traj['pilot_trajectory']:
                if len(point) >= 3:  # [timestamp, lat, lon]
                    pilot_coords.append([point[1], point[2]])  # [lat, lon]
            fixed_traj['pilot_trajectory'] = pilot_coords
        
        if 'vessel_trajectory' in fixed_traj and fixed_traj['vessel_trajectory']:
            vessel_coords = []
            for point in fixed_traj['vessel_trajectory']:
                if len(point) >= 3:  # [timestamp, lat, lon]
                    vessel_coords.append([point[1], point[2]])  # [lat, lon]
            fixed_traj['vessel_trajectory'] = vessel_coords
        
        fixed_trajectories[key] = fixed_traj
    
    print("✅ Data format issues fixed")
    return sessions, events, fixed_trajectories

def generate_fixed_html_visualization():
    """Generate the fixed HTML file with interactive visualization."""
    
    # Load and fix data
    sessions, events, trajectories = load_and_fix_data()
    
    # Convert data to JSON for JavaScript
    sessions_json = sessions.to_json(orient='records')
    events_json = events.to_json(orient='records')
    trajectories_json = json.dumps(trajectories)
    
    # Create HTML content with debugging and fixes
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Pilot Boat Assistance Sessions - Interactive Visualization (Fixed)</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
    <!-- Leaflet CSS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
    
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
        }}
        
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 15px;
            text-align: center;
        }}
        
        .controls {{
            background-color: #ecf0f1;
            padding: 15px;
            border-bottom: 1px solid #bdc3c7;
        }}
        
        .control-group {{
            margin-bottom: 10px;
        }}
        
        label {{
            font-weight: bold;
            margin-right: 10px;
        }}
        
        select {{
            padding: 5px;
            font-size: 14px;
            min-width: 400px;
        }}
        
        button {{
            background-color: #3498db;
            color: white;
            border: none;
            padding: 8px 15px;
            margin-left: 10px;
            cursor: pointer;
            border-radius: 3px;
        }}
        
        button:hover {{
            background-color: #2980b9;
        }}
        
        .info-panel {{
            background-color: #f8f9fa;
            padding: 10px;
            border-bottom: 1px solid #dee2e6;
            font-size: 14px;
        }}
        
        .debug-panel {{
            background-color: #fff3cd;
            padding: 10px;
            border-bottom: 1px solid #ffeaa7;
            font-size: 12px;
            font-family: monospace;
            max-height: 100px;
            overflow-y: auto;
        }}
        
        #map {{
            height: 600px;
            width: 100%;
        }}
        
        .legend {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: white;
            padding: 10px;
            border: 2px solid #ccc;
            border-radius: 5px;
            z-index: 1000;
            font-size: 12px;
        }}
        
        .legend h4 {{
            margin: 0 0 10px 0;
        }}
        
        .legend-item {{
            margin: 5px 0;
        }}
        
        .legend-color {{
            display: inline-block;
            width: 20px;
            height: 3px;
            margin-right: 5px;
            vertical-align: middle;
        }}
        
        .legend-marker {{
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
            vertical-align: middle;
        }}
        
        .stats-panel {{
            background-color: #e8f4fd;
            padding: 10px;
            margin: 10px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚢 Pilot Boat Assistance Sessions - Interactive Visualization (Fixed)</h1>
        <p>Explore pilot boat and vessel trajectories during assistance operations in Busan Port</p>
    </div>
    
    <div class="controls">
        <div class="control-group">
            <label for="sessionSelect">Select Assistance Session:</label>
            <select id="sessionSelect">
                <option value="">Choose a session...</option>
            </select>
            <button onclick="loadSession()">Load Session</button>
            <button onclick="toggleTrajectories()" id="toggleBtn">Hide Full Trajectories</button>
            <button onclick="fitToData()">Fit to Data</button>
            <button onclick="debugSession()">Debug Session</button>
        </div>
        <div class="control-group">
            <label for="trafficFilter">Filter by Traffic Direction:</label>
            <select id="trafficFilter" onchange="filterSessions()">
                <option value="">All Directions</option>
                <option value="inbound">Inbound</option>
                <option value="outbound">Outbound</option>
                <option value="mixed">Mixed</option>
                <option value="other">Other</option>
            </select>
            <label for="validationFilter">Filter by Validation:</label>
            <select id="validationFilter" onchange="filterSessions()">
                <option value="">All Types</option>
                <option value="course_aligned_speed_similar">Course Aligned & Speed Similar</option>
                <option value="boarding_operation">Boarding Operation</option>
                <option value="course_aligned">Course Aligned</option>
                <option value="speed_similar">Speed Similar</option>
            </select>
        </div>
    </div>
    
    <div class="debug-panel" id="debugPanel">
        Debug information will appear here...
    </div>
    
    <div class="info-panel" id="infoPanel">
        Select a session to view details and trajectories.
    </div>

    <div class="stats-panel" id="statsPanel">
        <h4>📊 Session Statistics</h4>
        <div id="sessionStats">Loading statistics...</div>
    </div>

    <div id="map"></div>
    
    <div class="legend">
        <h4>🗺️ Legend</h4>
        <div class="legend-item">
            <span class="legend-color" style="background-color: blue;"></span>
            Pilot Boat Trajectory
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background-color: red;"></span>
            Vessel Trajectory
        </div>
        <div class="legend-item">
            <span class="legend-marker" style="background-color: green;"></span>
            Course Aligned + Speed Similar
        </div>
        <div class="legend-item">
            <span class="legend-marker" style="background-color: orange;"></span>
            Boarding Operation
        </div>
        <div class="legend-item">
            <span class="legend-marker" style="background-color: purple;"></span>
            Other Validation
        </div>
        <div class="legend-item">
            <span class="legend-marker" style="background-color: yellow; border: 1px solid #ccc;"></span>
            Proximity Area
        </div>
    </div>
    
    <div class="stats-panel">
        <strong>📊 Dataset Statistics:</strong>
        Total Sessions: {len(sessions)} | 
        Total Proximity Events: {len(events)} | 
        Sessions with Trajectories: {len(trajectories)}
    </div>

    <!-- Leaflet JavaScript -->
    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
    
    <script>
        // Data from Python (fixed format)
        const sessionsData = {sessions_json};
        const eventsData = {events_json};
        const trajectoriesData = {trajectories_json};
        
        // Global variables
        let map;
        let currentLayers = [];
        let showFullTrajectories = true;
        let currentSessionIndex = null;
        
        // Debug function
        function debugLog(message) {{
            console.log(message);
            const debugPanel = document.getElementById('debugPanel');
            debugPanel.innerHTML += new Date().toLocaleTimeString() + ': ' + message + '<br>';
            debugPanel.scrollTop = debugPanel.scrollHeight;
        }}
        
        // Initialize map
        function initMap() {{
            debugLog('Initializing map...');
            
            // Busan Port coordinates
            const busanLat = 35.1040;
            const busanLon = 129.0403;
            
            map = L.map('map').setView([busanLat, busanLon], 12);
            
            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                attribution: '© OpenStreetMap contributors'
            }}).addTo(map);
            
            // Add Busan Port marker
            L.marker([busanLat, busanLon], {{
                icon: L.divIcon({{
                    className: 'port-marker',
                    html: '<div style="background-color: green; width: 15px; height: 15px; border-radius: 50%; border: 3px solid white; box-shadow: 0 0 5px rgba(0,0,0,0.5);"></div>',
                    iconSize: [21, 21]
                }})
            }})
                .addTo(map)
                .bindPopup('<b>🏭 Busan Port</b><br>Major container port in South Korea')
                .openPopup();
            
            debugLog('Map initialized successfully');
        }}
        
        // Populate session dropdown
        function populateSessionDropdown() {{
            debugLog('Populating session dropdown...');
            const select = document.getElementById('sessionSelect');
            
            sessionsData.forEach((session, index) => {{
                const option = document.createElement('option');
                option.value = index;
                
                const startTime = new Date(session.start_time).toLocaleString();
                const trafficDir = session.primary_traffic_direction || 'unknown';
                
                option.text = `Session ${{index + 1}}: Pilot ${{session.pilot_mmsi}} + Vessel ${{session.vessel_mmsi}} - ${{startTime}} (${{session.duration_minutes.toFixed(1)}}min, ${{trafficDir}})`;
                
                select.appendChild(option);
            }});
            
            debugLog(`Populated dropdown with ${{sessionsData.length}} sessions`);
        }}
        
        // Clear current layers
        function clearLayers() {{
            debugLog(`Clearing ${{currentLayers.length}} layers`);
            currentLayers.forEach(layer => {{
                map.removeLayer(layer);
            }});
            currentLayers = [];
        }}
        
        // Load selected session
        function loadSession() {{
            const sessionIndex = document.getElementById('sessionSelect').value;
            if (!sessionIndex) {{
                debugLog('No session selected');
                return;
            }}
            
            debugLog(`Loading session ${{sessionIndex}}...`);
            currentSessionIndex = sessionIndex;
            const session = sessionsData[sessionIndex];
            clearLayers();
            
            // Update info panel
            updateInfoPanel(session);
            
            // Find trajectory data
            const trajectoryKey = findTrajectoryKey(session);
            if (!trajectoryKey) {{
                debugLog(`No trajectory data found for session ${{sessionIndex}}`);
                alert('⚠️ No trajectory data found for this session');
                return;
            }}
            
            debugLog(`Found trajectory key: ${{trajectoryKey}}`);
            const trajectoryData = trajectoriesData[trajectoryKey];
            
            // Get trajectories (already in correct format)
            const pilotTrajectory = trajectoryData.pilot_trajectory || [];
            const vesselTrajectory = trajectoryData.vessel_trajectory || [];
            
            debugLog(`Pilot trajectory points: ${{pilotTrajectory.length}}`);
            debugLog(`Vessel trajectory points: ${{vesselTrajectory.length}}`);
            
            // Add trajectories to map
            if (showFullTrajectories) {{
                addTrajectoriesToMap(pilotTrajectory, vesselTrajectory, session);
            }}
            
            // Add proximity events
            addProximityEventsToMap(session);
            
            // Fit map to show all data
            fitToData();
            
            debugLog(`Session ${{sessionIndex}} loaded successfully`);
        }}
        
        // Find trajectory key for session
        function findTrajectoryKey(session) {{
            debugLog(`Looking for trajectory for pilot ${{session.pilot_mmsi}}, vessel ${{session.vessel_mmsi}}, start ${{session.start_time}}`);
            
            for (const key in trajectoriesData) {{
                const traj = trajectoriesData[key];
                if (traj.pilot_mmsi == session.pilot_mmsi && 
                    traj.vessel_mmsi == session.vessel_mmsi &&
                    traj.start_time === session.start_time) {{
                    return key;
                }}
            }}
            return null;
        }}
        
        // Add trajectories to map
        function addTrajectoriesToMap(pilotTrajectory, vesselTrajectory, session) {{
            debugLog('Adding trajectories to map...');
            
            // Add pilot trajectory
            if (pilotTrajectory && pilotTrajectory.length > 1) {{
                debugLog(`Adding pilot trajectory with ${{pilotTrajectory.length}} points`);
                
                const pilotLine = L.polyline(pilotTrajectory, {{
                    color: 'blue',
                    weight: 4,
                    opacity: 0.8,
                    dashArray: '5, 5'
                }}).bindPopup(`🚤 Pilot Boat ${{session.pilot_mmsi}} Trajectory<br>Points: ${{pilotTrajectory.length}}`);
                
                currentLayers.push(pilotLine);
                pilotLine.addTo(map);
                
                // Add start/end markers for pilot
                const pilotStart = L.marker(pilotTrajectory[0], {{
                    icon: L.divIcon({{
                        className: 'custom-marker',
                        html: '<div style="background-color: blue; width: 12px; height: 12px; border-radius: 50%; border: 2px solid white; box-shadow: 0 0 3px rgba(0,0,0,0.5);"></div>',
                        iconSize: [16, 16]
                    }})
                }}).bindPopup(`🚤 Pilot Start<br>${{session.start_time}}`);
                
                const pilotEnd = L.marker(pilotTrajectory[pilotTrajectory.length - 1], {{
                    icon: L.divIcon({{
                        className: 'custom-marker',
                        html: '<div style="background-color: darkblue; width: 12px; height: 12px; border-radius: 50%; border: 2px solid white; box-shadow: 0 0 3px rgba(0,0,0,0.5);"></div>',
                        iconSize: [16, 16]
                    }})
                }}).bindPopup(`🚤 Pilot End<br>${{session.end_time}}`);
                
                currentLayers.push(pilotStart, pilotEnd);
                pilotStart.addTo(map);
                pilotEnd.addTo(map);
            }}
            
            // Add vessel trajectory
            if (vesselTrajectory && vesselTrajectory.length > 1) {{
                debugLog(`Adding vessel trajectory with ${{vesselTrajectory.length}} points`);
                
                const vesselLine = L.polyline(vesselTrajectory, {{
                    color: 'red',
                    weight: 4,
                    opacity: 0.8
                }}).bindPopup(`🚢 Vessel ${{session.vessel_mmsi}} Trajectory<br>Points: ${{vesselTrajectory.length}}`);
                
                currentLayers.push(vesselLine);
                vesselLine.addTo(map);
                
                // Add start/end markers for vessel
                const vesselStart = L.marker(vesselTrajectory[0], {{
                    icon: L.divIcon({{
                        className: 'custom-marker',
                        html: '<div style="background-color: red; width: 12px; height: 12px; border-radius: 50%; border: 2px solid white; box-shadow: 0 0 3px rgba(0,0,0,0.5);"></div>',
                        iconSize: [16, 16]
                    }})
                }}).bindPopup(`🚢 Vessel Start<br>${{session.start_time}}`);
                
                const vesselEnd = L.marker(vesselTrajectory[vesselTrajectory.length - 1], {{
                    icon: L.divIcon({{
                        className: 'custom-marker',
                        html: '<div style="background-color: darkred; width: 12px; height: 12px; border-radius: 50%; border: 2px solid white; box-shadow: 0 0 3px rgba(0,0,0,0.5);"></div>',
                        iconSize: [16, 16]
                    }})
                }}).bindPopup(`🚢 Vessel End<br>${{session.end_time}}`);
                
                currentLayers.push(vesselStart, vesselEnd);
                vesselStart.addTo(map);
                vesselEnd.addTo(map);
            }}
        }}
        
        // Add proximity events to map
        function addProximityEventsToMap(session) {{
            debugLog('Adding proximity events to map...');
            
            const sessionEvents = eventsData.filter(event => 
                event.pilot_mmsi == session.pilot_mmsi &&
                event.vessel_mmsi == session.vessel_mmsi &&
                event.timestamp >= session.start_time &&
                event.timestamp <= session.end_time
            );
            
            debugLog(`Found ${{sessionEvents.length}} proximity events for session`);
            
            sessionEvents.forEach((event, index) => {{
                // Determine marker color and icon based on validation reason
                let color = 'purple';
                let icon = '❓';
                if (event.validation_reason === 'course_aligned_speed_similar') {{
                    color = 'green';
                    icon = '✅';
                }} else if (event.validation_reason === 'boarding_operation') {{
                    color = 'orange';
                    icon = '🔄';
                }}
                
                // Create popup content
                const popupContent = `
                    <b>${{icon}} Proximity Event #${{index + 1}}</b><br>
                    <strong>Time:</strong> ${{new Date(event.timestamp).toLocaleString()}}<br>
                    <strong>Distance:</strong> ${{event.distance.toFixed(1)}}m<br>
                    <strong>Pilot Speed:</strong> ${{event.pilot_sog.toFixed(1)}} knots<br>
                    <strong>Vessel Speed:</strong> ${{event.vessel_sog.toFixed(1)}} knots<br>
                    <strong>Course Aligned:</strong> ${{event.is_course_aligned ? '✅' : '❌'}}<br>
                    <strong>Speed Similar:</strong> ${{event.is_speed_similar ? '✅' : '❌'}}<br>
                    <strong>Validation:</strong> ${{event.validation_reason}}<br>
                    <strong>Traffic Direction:</strong> ${{event.traffic_direction}}
                `;
                
                // Add proximity marker
                const marker = L.marker([event.vessel_lat, event.vessel_lon], {{
                    icon: L.divIcon({{
                        className: 'proximity-marker',
                        html: `<div style="background-color: ${{color}}; width: 14px; height: 14px; border-radius: 50%; border: 2px solid white; box-shadow: 0 0 5px rgba(0,0,0,0.7); display: flex; align-items: center; justify-content: center; font-size: 8px; color: white; font-weight: bold;">${{index + 1}}</div>`,
                        iconSize: [18, 18]
                    }})
                }}).bindPopup(popupContent);
                
                currentLayers.push(marker);
                marker.addTo(map);
                
                // Add proximity circle
                const circle = L.circle([event.vessel_lat, event.vessel_lon], {{
                    radius: event.distance,
                    color: 'yellow',
                    weight: 2,
                    opacity: 0.6,
                    fillOpacity: 0.1,
                    dashArray: '3, 3'
                }}).bindPopup(`📏 Proximity Zone<br>Distance: ${{event.distance.toFixed(1)}}m`);
                
                currentLayers.push(circle);
                circle.addTo(map);
            }});
        }}
        
        // Update info panel
        function updateInfoPanel(session) {{
            const panel = document.getElementById('infoPanel');
            const startTime = new Date(session.start_time).toLocaleString();
            const endTime = new Date(session.end_time).toLocaleString();

            panel.innerHTML = `
                <strong>📋 Session Details:</strong><br>
                🚤 <strong>Pilot MMSI:</strong> ${{session.pilot_mmsi}} |
                🚢 <strong>Vessel MMSI:</strong> ${{session.vessel_mmsi}}<br>
                ⏱️ <strong>Duration:</strong> ${{session.duration_minutes.toFixed(1)}} minutes |
                📍 <strong>Observations:</strong> ${{session.num_observations}}<br>
                🧭 <strong>Traffic Direction:</strong> ${{session.primary_traffic_direction || 'unknown'}} |
                ✅ <strong>Validation:</strong> ${{session.primary_validation_reason || 'unknown'}}<br>
                📏 <strong>Distance Range:</strong> ${{session.min_distance.toFixed(1)}}m - ${{session.max_distance.toFixed(1)}}m |
                📊 <strong>Avg Distance:</strong> ${{session.avg_distance.toFixed(1)}}m<br>
                🧭 <strong>Course Alignment:</strong> ${{(session.course_alignment_ratio * 100).toFixed(1)}}% |
                🚀 <strong>Speed Similarity:</strong> ${{(session.speed_similarity_ratio * 100).toFixed(1)}}%<br>
                ⚓ <strong>Boarding Operations:</strong> ${{(session.boarding_operation_ratio * 100).toFixed(1)}}% |
                🎯 <strong>Trajectory Points:</strong> P:${{session.pilot_trajectory_points || 0}} V:${{session.vessel_trajectory_points || 0}}<br>
                🕐 <strong>Start:</strong> ${{startTime}}<br>
                🕑 <strong>End:</strong> ${{endTime}}
            `;
        }}
        
        // Toggle trajectory display
        function toggleTrajectories() {{
            showFullTrajectories = !showFullTrajectories;
            const button = document.getElementById('toggleBtn');
            button.textContent = showFullTrajectories ? 'Hide Full Trajectories' : 'Show Full Trajectories';
            
            // Reload current session if one is selected
            if (currentSessionIndex !== null) {{
                loadSession();
            }}
        }}
        
        // Fit map to current data
        function fitToData() {{
            if (currentLayers.length > 0) {{
                const group = new L.featureGroup(currentLayers);
                map.fitBounds(group.getBounds().pad(0.1));
                debugLog('Fitted map to data bounds');
            }}
        }}
        
        // Debug current session
        function debugSession() {{
            if (currentSessionIndex !== null) {{
                const session = sessionsData[currentSessionIndex];
                debugLog(`Current session: ${{JSON.stringify(session, null, 2)}}`);
                
                const trajectoryKey = findTrajectoryKey(session);
                if (trajectoryKey) {{
                    const traj = trajectoriesData[trajectoryKey];
                    debugLog(`Trajectory data: ${{JSON.stringify(traj, null, 2)}}`);
                }}
            }} else {{
                debugLog('No session currently selected');
            }}
        }}

        // Filter sessions based on selected criteria
        function filterSessions() {{
            const trafficFilter = document.getElementById('trafficFilter').value;
            const validationFilter = document.getElementById('validationFilter').value;

            const sessionSelect = document.getElementById('sessionSelect');
            const allOptions = sessionSelect.querySelectorAll('option');

            // Hide all options first
            allOptions.forEach(option => {{
                if (option.value === '') return; // Keep the default option

                const sessionIndex = parseInt(option.value);
                const session = sessionsData[sessionIndex];

                let showOption = true;

                // Apply traffic direction filter
                if (trafficFilter && session.primary_traffic_direction !== trafficFilter) {{
                    showOption = false;
                }}

                // Apply validation filter
                if (validationFilter && session.primary_validation_reason !== validationFilter) {{
                    showOption = false;
                }}

                option.style.display = showOption ? 'block' : 'none';
            }});

            updateSessionStatistics();
        }}

        // Update session statistics
        function updateSessionStatistics() {{
            const trafficFilter = document.getElementById('trafficFilter').value;
            const validationFilter = document.getElementById('validationFilter').value;

            let filteredSessions = sessionsData;

            if (trafficFilter) {{
                filteredSessions = filteredSessions.filter(s => s.primary_traffic_direction === trafficFilter);
            }}

            if (validationFilter) {{
                filteredSessions = filteredSessions.filter(s => s.primary_validation_reason === validationFilter);
            }}

            // Calculate statistics
            const totalSessions = filteredSessions.length;
            const avgDuration = filteredSessions.reduce((sum, s) => sum + s.duration_minutes, 0) / totalSessions;
            const avgDistance = filteredSessions.reduce((sum, s) => sum + s.avg_distance, 0) / totalSessions;

            // Count by traffic direction
            const trafficCounts = {{}};
            filteredSessions.forEach(s => {{
                trafficCounts[s.primary_traffic_direction] = (trafficCounts[s.primary_traffic_direction] || 0) + 1;
            }});

            // Count by validation reason
            const validationCounts = {{}};
            filteredSessions.forEach(s => {{
                validationCounts[s.primary_validation_reason] = (validationCounts[s.primary_validation_reason] || 0) + 1;
            }});

            const statsHtml = `
                <strong>Filtered Sessions:</strong> ${{totalSessions}}<br>
                <strong>Average Duration:</strong> ${{avgDuration.toFixed(1)}} minutes<br>
                <strong>Average Distance:</strong> ${{avgDistance.toFixed(1)}} meters<br>
                <br>
                <strong>Traffic Direction:</strong><br>
                ${{Object.entries(trafficCounts).map(([key, value]) => `&nbsp;&nbsp;${{key}}: ${{value}}`).join('<br>')}}<br>
                <br>
                <strong>Validation Reasons:</strong><br>
                ${{Object.entries(validationCounts).map(([key, value]) => `&nbsp;&nbsp;${{key}}: ${{value}}`).join('<br>')}}
            `;

            document.getElementById('sessionStats').innerHTML = statsHtml;
        }}

        // Initialize everything when page loads
        document.addEventListener('DOMContentLoaded', function() {{
            debugLog('Page loaded, initializing...');
            initMap();
            populateSessionDropdown();
            updateSessionStatistics();

            // Load first session by default if available
            if (sessionsData.length > 0) {{
                document.getElementById('sessionSelect').value = 0;
                loadSession();
            }}
        }});
    </script>
</body>
</html>
"""
    
    # Save HTML file
    with open('pilot_boat_visualization_fixed.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("✅ Created ENHANCED interactive visualization: pilot_boat_visualization_fixed.html")
    print(f"📊 Enhanced visualization includes:")
    print(f"   - {len(sessions)} assistance sessions")
    print(f"   - {len(events)} proximity events")
    print(f"   - {len(trajectories)} trajectory datasets")
    print(f"🔧 Key fixes and enhancements:")
    print(f"   - Fixed trajectory coordinate format ([lat, lon] instead of [timestamp, lat, lon])")
    print(f"   - Fixed MMSI data type consistency (int instead of float)")
    print(f"   - Fixed start_time format matching")
    print(f"   - Added comprehensive debugging")
    print(f"   - Added traffic direction and validation filtering")
    print(f"   - Enhanced session details with new metrics")
    print(f"   - Added real-time session statistics")
    print(f"   - Improved info panel with detailed session data")
    print(f"🌐 Open 'pilot_boat_visualization_fixed.html' in your web browser!")
    
    return html_content

if __name__ == "__main__":
    generate_fixed_html_visualization()
