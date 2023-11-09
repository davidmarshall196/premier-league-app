import streamlit as st
import pandas as pd
import pydeck as pdk

# Load the CSV file
file_path = '/Users/david@inawisdom.com/Documents/Training/premier_league/data/stadiums-with-GPS-coordinates.csv'
stadiums_df = pd.read_csv(file_path, encoding="ISO-8859-1")

# Dropdown for selecting a stadium
selected_stadium = st.selectbox('Select a stadium:', stadiums_df['Stadium'])

# Get the selected stadium's information
stadium_info = stadiums_df[stadiums_df['Stadium'] == selected_stadium].iloc[0]

# Display stadium information
st.write(f"Stadium: {stadium_info['Stadium']}")
st.write(f"Team: {stadium_info['Team']}")
st.write(f"City: {stadium_info['City']}")
st.write(f"Capacity: {stadium_info['Capacity']}")

# Plot the selected stadium on a map
st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/satellite-streets-v11',
    initial_view_state=pdk.ViewState(
        latitude=stadium_info['Latitude'],
        longitude=stadium_info['Longitude'],
        zoom=14
    ),
))