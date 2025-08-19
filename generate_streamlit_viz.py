import streamlit as st
import os
import pandas as pd
import ast

import folium
from streamlit_folium import st_folium

# Very useful DataCamp article
# https://www.datacamp.com/tutorial/streamlit

#region Examples
# Some sample UI features
# st.title("This is the app title")
# st.header("This is the header")
# st.markdown("<i>This is the markdown</i>")
# st.subheader("This is the subheader")
# st.caption("This is the caption")
# st.code("x = 2021")
# st.latex(r''' a+a r^1+a r^2+a r^3 ''')

# Some sample media controls
# st.image("kid.jpg", caption="A kid playing")
# st.audio("audio.mp3")
# st.video("video.mp4")

# Some sample widgets
# st.checkbox('Yes')
# st.button('Click Me')
# st.radio('Pick your gender', ['Male', 'Female'])
# st.selectbox('Pick a fruit', ['Apple', 'Banana', 'Orange'])
# st.multiselect('Choose a planet', ['Jupiter', 'Mars', 'Neptune'])
# st.select_slider('Pick a mark', ['Bad', 'Good', 'Excellent'])
# st.slider('Pick a number', 0, 50)

# Input controls
# st.number_input('Pick a number', 0, 10)
# st.text_input('Email address')
# st.date_input('Traveling date')
# st.time_input('School time')
# st.text_area('Description')
# st.file_uploader('Upload a photo')
# st.color_picker('Choose your favorite color')

# Warning and and information
# st.success("You did it!")
# st.error("Error occurred")
# st.warning("This is a warning")
# st.info("It's easy to build a Streamlit app")
# st.exception(RuntimeError("RuntimeError exception"))

# Sidebar containers
# st.sidebar.title("Sidebar Title")
# st.sidebar.markdown("This is the sidebar content")

# Hidden container for organization purposes
# with st.container():    st.write("This is inside the container")
#endregion

st.set_page_config(layout="wide")
# Inject custom CSS to change sidebar width
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            width: 400px !important;   /* Set your desired width */
        }
        [data-testid="stSidebarContent"] {
            width: 400px !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# Coordinates for Busan Port (approximate center)
BUSAN_PORT_COORDS = [35.082887, 128.832212]

# Zoom level for the map (adjust as needed)
DEFAULT_ZOOM = 14



# Full path of the target data folder
# Month label is formatted like e.g., -> 202309 (YYYYMM)
def load_data_for_selected_period(full_data_path: str, period_folder_name: str):

    # period_folder_name is like 'month_202309'
    if '_' in period_folder_name:
        label = period_folder_name.split('_', 1)[1]
    else:
        label = period_folder_name

    data_file_path = os.path.join(full_data_path, f"pilot_boat_assistance_sessions_{label}.csv")
    if not os.path.exists(data_file_path):
        st.warning(f"Pilot Boat Assistance sessions .csv not found at {data_file_path}")

    low_level_proximity_file_path = os.path.join(full_data_path, f"pilot_boat_proximity_events_{label}.csv")
    if not os.path.exists(low_level_proximity_file_path):
        st.warning(f"Pilot Boat Proximity .csv not found at {low_level_proximity_file_path}")

    loaded_data_df = pd.read_csv(data_file_path)
    # Cast timestamp to proper datetime
    loaded_data_df["start_time"] = pd.to_datetime(loaded_data_df["start_time"])
    loaded_data_df["end_time"] = pd.to_datetime(loaded_data_df["end_time"])

    print(f"Session data loaded at {data_file_path}")
    st.success(f"Session data loaded at {data_file_path}")

    loaded_proximity_data_df = pd.read_csv(low_level_proximity_file_path)
    # Cast timestamp to proper datetime
    loaded_proximity_data_df["timestamp"] = pd.to_datetime(loaded_proximity_data_df["timestamp"])
    print(f"Proximity data loaded {low_level_proximity_file_path}")
    st.success(f"Proximity data loaded from {low_level_proximity_file_path}")

    # compute some basic stats
    stats = {
        "total_sessions" : loaded_data_df.shape[0],
        "inbounds" : loaded_data_df[loaded_data_df["primary_traffic_direction"] == 'inbound'].shape[0],
        "outbounds" : loaded_data_df[loaded_data_df["primary_traffic_direction"] == 'outbound'].shape[0],
        "others" : loaded_data_df[loaded_data_df["primary_traffic_direction"] == 'other'].shape[0],
        "mixed" : loaded_data_df[loaded_data_df["primary_traffic_direction"] == 'mixed'].shape[0],
    }

    return loaded_data_df, stats, loaded_proximity_data_df

def get_formatted_sessions_with_index(loaded_df : pd.DataFrame):
    index_session_names = []
    for index, row in loaded_df.iterrows():
        formatted_name = f"{index} | Pilot {row['pilot_mmsi']} + Ship {row['vessel_mmsi']} {row['start_time']} ({row['duration_minutes']}, {row['primary_traffic_direction']})"
        index_session_names.append((index, formatted_name))        
    return index_session_names

root_results_path = "./results/monthly_analysis"


if not os.path.exists(root_results_path):
    err = f"Results path not found at  {root_results_path}"
    print(err)
    st.exception(err)
    exit()

st.header("Maritime Pilot Boat and Vessel Interaction Optimization Visualization")

periods_available = sorted(os.listdir(root_results_path))

if len(periods_available) == 0:
    st.warning(f"No data found! at {root_results_path}")
    exit()

st.sidebar.markdown("<br>", unsafe_allow_html=True)  # One line of space
selected_period = st.sidebar.selectbox("Please Select A Month", periods_available, index=0)
st.sidebar.write(f"Available interactions for <b>{len(periods_available)} months</b>", unsafe_allow_html=True)


loaded_df, stats, proximity_df = load_data_for_selected_period(os.path.join(root_results_path, selected_period), selected_period)

st.sidebar.markdown("<br>", unsafe_allow_html=True)  # One line of space
st.sidebar.write(f"Available Interactions in <i>{selected_period}</i> ->  <b>{stats['total_sessions']}</b>", unsafe_allow_html=True)

map = folium.Map(location=BUSAN_PORT_COORDS, zoom_start=DEFAULT_ZOOM)

if loaded_df is not None:
 st.sidebar.write(
        f"Inbound sessions -> <b>{stats['inbounds']}</b><br>"
        f"Outbound sessions -> <b>{stats['outbounds']}</b><br>"
        f"Other sessions -> <b>{stats['others']}</b> <br>"
        f"Mixed sessions -> <b>{stats['mixed']}</b>  <i>*sessions where the algorithm detected conflicting movement</i>",
        unsafe_allow_html=True 
    )
 
 session_choice = st.sidebar.radio("Select Session Type", ["All", "inbound", "outbound", "other", "mixed"], index=0)
 
 if session_choice.lower() != "all":
     loaded_df = loaded_df[loaded_df["primary_traffic_direction"] == session_choice]

 index_session_names = get_formatted_sessions_with_index(loaded_df)
 indexes_only = [x[0] for x in index_session_names]
 sessions_only = [x[1] for x in index_session_names]

  

 selected_session = st.sidebar.selectbox("Select a session", sessions_only, index=0)

 if selected_session is not None:
     selected_session_index = selected_session.split('|')[0]
     p_mmsi = loaded_df.loc[int(selected_session_index), 'pilot_mmsi']
     v_mmsi = loaded_df.loc[int(selected_session_index), 'vessel_mmsi']
     duration = loaded_df.loc[(int(selected_session_index), 'duration_minutes')]
     observations = loaded_df.loc[(int(selected_session_index), 'num_observations')]
     direction = loaded_df.loc[int(selected_session_index), "primary_traffic_direction"]

     start_time = loaded_df.loc[int(selected_session_index), 'start_time']
     end_time = loaded_df.loc[int(selected_session_index), 'end_time']
     st.sidebar.write(
         f"Direction -> {direction} <br>"
        f"Pilot <b>{p_mmsi}</b> | Vessel <b>{v_mmsi}</b> <br>"
        f"Duration <b>{duration} min</b> | Observations <b>{observations}</b> <br>"
        f"Start -> <b>{start_time}</b> <br>"
        f"End -> <b>{end_time}</b> <br>"
        ,
        unsafe_allow_html=True
     )

     # Load the trajectories on the map
     v_ext_traj = ast.literal_eval(loaded_df.loc[int(selected_session_index), 'vessel_extended_trajectory'])
     v_ext_lat_lng = [[t['latitude'], t['longitude']] for t in v_ext_traj]
     
     p_ext_traj = ast.literal_eval(loaded_df.loc[int(selected_session_index), 'pilot_extended_trajectory'])
     p_ext_lat_lng = [[t['latitude'], t['longitude']] for t in p_ext_traj]

     # Plot Vessel trajectory data
     folium.Marker(
            location=v_ext_lat_lng[0],
            popup=f"Vessel Start {v_ext_traj[0]['timestamp']}<br>COG {v_ext_traj[0]['cog']}<br> SOG {v_ext_traj[0]['sog']}",
            icon=folium.Icon(color="red", icon="info-sign")
     ).add_to(map)
     folium.PolyLine(locations=v_ext_lat_lng, color="red", weight=2, opacity=1, tooltip="Vessel").add_to(map)
     folium.Marker(
            location=v_ext_lat_lng[len(v_ext_lat_lng)-1],
            popup=f"Vessel End {v_ext_traj[len(v_ext_lat_lng)-1]['timestamp']}<br>COG {v_ext_traj[len(v_ext_lat_lng)-1]['cog']}<br> SOG {v_ext_traj[len(v_ext_lat_lng)-1]['sog']}",
            icon=folium.Icon(color="red", icon="info-sign")
     ).add_to(map)
     
     # Also plot white markets for extra meta data
     for idx, traj in enumerate(v_ext_traj):
        html = f'''
            <div style="
                background-color: white; 
                color: white;
                border-radius: 50%;
                width: 2px;
                height: 2px;
                text-align: center;
                line-height: 1px;
                font-size: 1px;
                font-weight: bold;"> 
                {idx}
            </div>
            '''
        idx += 1
        tooltip = f"""
            <div style="font-size:12px; width: 150px">
                Vessel AIS Data <br>
                Time {traj['timestamp']}<br>
                COG {traj['cog']}<br>
                SOG {traj['sog']}<br>
            </div> 
            """
        folium.CircleMarker(
        location=[traj['latitude'], traj['longitude']],
        raidus=1,
        color='white',
        fill=True,

        popup=tooltip,
        icon=folium.DivIcon(html=html)
        ).add_to(map)





    # Plot Pilot trajectory data
     folium.Marker(
            location=p_ext_lat_lng[0],
            popup=f"Pilot Start {p_ext_traj[0]['timestamp']}<br>COG {p_ext_traj[0]['cog']}<br> SOG {p_ext_traj[0]['sog']}",
            icon=folium.Icon(color="blue", icon="info-sign")
     ).add_to(map)
     folium.PolyLine(locations=p_ext_lat_lng, color="blue", weight=2, opacity=1, tooltip="Vessel").add_to(map)
     folium.Marker(
            location=p_ext_lat_lng[len(p_ext_lat_lng)-1],
            popup=f"Pilot End {p_ext_traj[len(p_ext_lat_lng)-1]['timestamp']}<br>COG {p_ext_traj[len(p_ext_lat_lng)-1]['cog']}<br> SOG {p_ext_traj[len(p_ext_lat_lng)-1]['sog']}",
            icon=folium.Icon(color="blue", icon="info-sign")
     ).add_to(map)
     # Also plot white markets for extra meta data
     for idx, traj in enumerate(p_ext_traj):
        html = f'''
            <div style="
                background-color: white; 
                color: white;
                border-radius: 50%;
                width: 2px;
                height: 2px;
                text-align: center;
                line-height: 1px;
                font-size: 1px;
                font-weight: bold;"> 
                {idx}
            </div>
            '''
        idx += 1
        tooltip = f"""
            <div style="font-size:12px; width: 150px">
                Pilot AIS Data <br>
                Time {traj['timestamp']}<br>
                COG {traj['cog']}<br>
                SOG {traj['sog']}<br>
            </div> 
            """
        folium.CircleMarker(
        location=[traj['latitude'], traj['longitude']],
        raidus=1,
        color='white',
        fill=True,

        popup=tooltip,
        icon=folium.DivIcon(html=html)
        ).add_to(map)

     # Plot proximity events
     filtered_proximity_events = proximity_df[(proximity_df['pilot_mmsi'] == p_mmsi) & (proximity_df['vessel_mmsi'] == v_mmsi) & (proximity_df['timestamp'] >= start_time) & (proximity_df['timestamp'] <= end_time)]
     filtered_proximity_events = filtered_proximity_events.sort_values(by="timestamp", ascending=True)
     print(filtered_proximity_events.shape[0])
     
     idx = 0
     for i, row in filtered_proximity_events.iterrows():
          v_lat = row['vessel_lat']
          v_lon = row['vessel_lon']
          print(idx,[v_lat, v_lon])
          html = f'''
            <div style="
                background-color: {'green' if (row['is_course_aligned'] == True and row['is_speed_similar'] == True) else 'purple'}; 
                color: white;
                border-radius: 50%;
                width: 24px;
                height: 24px;
                text-align: center;
                line-height: 24px;
                font-size: 14px;
                font-weight: bold;">
                {idx} 
            </div>
            '''
          idx += 1
          tooltip = f"""
           <div style="font-size:12px; width: 150px">
             Time {row['timestamp']}<br>
             Distance {round(row['distance'], 2)}<br>
             Pilot Speed {round(row['pilot_sog'], 2)} knots<br>
             Vessel Speed {round(row['vessel_sog'],2 )} knots<br>
             Course Aligned {'✅' if row['is_course_aligned'] == True else '❌'}<br>
             Speed Similar {'✅' if row['is_speed_similar'] == True else '❌'}<br>
           </div> 
          """
          folium.Marker(
            location=[v_lat, v_lon],
            popup=tooltip,
            icon=folium.DivIcon(html=html)
          ).add_to(map)

# Adding marker for Busan port
folium.Marker(
    location=BUSAN_PORT_COORDS,
    popup="<b> BUSAN PORT </b>",
    icon=folium.Icon(color="darkblue", icon="ship", prefix="fa")
).add_to(map)
# --- Legend Creation ---
st.markdown("""
<style>
    .legend-container {
        border: 1px solid #ddd;
        padding: 10px;
        border-radius: 5px;
        background-color: #f9f9f9;
        font-family: Arial, sans-serif;
        font-size: 14px;
        max-width: 300px; /* Adjust width as needed */
    }
    .legend-title {
        font-weight: bold;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
    }
    .legend-title img {
        margin-right: 5px;
        height: 20px; /* Adjust icon size */
        width: 20px;
    }
    .legend-item {
        display: flex;
        align-items: center;
        margin-bottom: 5px;
    }
    .legend-color-line {
        width: 30px; /* Length of the line */
        height: 4px; /* Thickness of the line */
        margin-right: 10px;
        border-radius: 2px; /* Slightly rounded ends */
    }
    /* Specific colors (only standard versions remain) */
    .color-pilot-standard { background-color: #007bff; } /* Standard blue */
    .color-vessel-standard { background-color: #dc3545; } /* Standard red */
</style>

<div class="legend-container">
    <div class="legend-title">
        🗺️ Legend
    </div>
    <div class="legend-item">
        <div class="legend-color-line color-pilot-standard"></div>
        Pilot Boat
    </div>
    <div class="legend-item">
        <div class="legend-color-line color-vessel-standard"></div>
        Vessel
    </div>
</div>
""", unsafe_allow_html=True)
# Display the map in Streamlit
st_folium(map, width=1000, height=800)

# st.sidebar.header("Map Controls")
# selected_zoom = st.sidebar.slider("Zoom Level", 10, 15, DEFAULT_ZOOM)
# map.options["zoom"] = selected_zoom # Update map zoom (requires re-render of map)