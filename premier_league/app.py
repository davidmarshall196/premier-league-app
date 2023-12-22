import streamlit as st
import matplotlib.pyplot as plt
import shap
import pydeck as pdk
import data_extraction
import visualisations
import constants
import s3_helpers
import logger_config

# source env_premier_league/bin/activate
# streamlit run premier_league/streamlit_app.py
logger_config.logger.info("Streamlit: app starting")

# Grab data
transformed_data = s3_helpers.grab_data_s3(constants.PREDICTIONS_LOCATION)

# Stadium data
stadium_data = s3_helpers.grab_data_s3(constants.STADIUM_DATA_LOCATION)

# Model performance
model_performance = s3_helpers.grab_data_s3(constants.MODEL_PERFORMANCE_LOCATION)

# Home Team
regressor1 = s3_helpers.load_transformer_s3_pickle(
    constants.HOME_MODEL_PREFIX, is_transformer=False
)

# Away Team
regressor2 = s3_helpers.load_transformer_s3_pickle(
    constants.AWAY_MODEL_PREFIX, is_transformer=False
)

# Extract current fixtures
current_fixtures = data_extraction.extract_current_fixtures(transformed_data)

# Add day
current_fixtures["date_format"] = current_fixtures["MatchDayDate"].apply(
    lambda x: visualisations.get_date_suffix(x)
)
current_fixtures["full_date"] = (
    current_fixtures["MatchDayDay"]
    + " "
    + current_fixtures["date_format"]
    + " "
    + current_fixtures["MatchDayMonth"]
)

# Extract next fixtures
pred_df = (
    current_fixtures[
        [
            "HomeTeam",
            "Home Prediction",
            "Away Prediction",
            "AwayTeam",
            "full_date",
            "Fixture Time",
            "Location",
        ]
    ]
    .reset_index(drop=True)
    .head(10)
)
pred_df.columns = constants.PRED_DF_COL_NAMES
pred_df.index = pred_df.index + 1


# Wide layout
st.set_page_config(layout="wide")

# Define the custom CSS
custom_css = """
<style>
    .sidebar .sidebar-content {
        width: 600px;
    }

    .reportview-container .main .block-container {
        padding-top: {1}rem;
        padding-bottom: {1}rem;
        padding-left: {1}rem;
        padding-right: {1}rem;
    }
    .appview-container .main .block-container{{
        padding-top: {1}rem;
        padding-bottom: {1}rem;
        padding-left: {1}rem;
        padding-right: {1}rem;
    }}

</style>
"""

# Inject the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Title
st.title("Premier League Prediction Dashboard")


# Create two columns
col1, col2 = st.columns(2)

# Display previous results in the left column
with col1:
    # Left panel for overall information
    st.header("Upcoming Fixtures Information")

    # Table of the predictions
    st.subheader("Predictions:")
    st.dataframe(pred_df)

    # Table
    html_code = """
    <div class="fwp-embed" data-url="premier-league/league-table"></div>
    <script src="https://www.footballwebpages.co.uk/embed.js" defer></script>
    """
    st.subheader("Premier League Table")
    st.components.v1.html(html_code, height=300)

    # Shap summary
    shap_values, features = visualisations.get_shap_values(transformed_data, regressor1)
    st.subheader("SHAP Summary: Home Team")
    shap.initjs()
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, transformed_data[regressor1.feature_names_], show=True
    )
    plt.title("Prediction for Home Team Goals")
    plt.xlabel("Shap Impact")
    st.pyplot(plt)

    # Shap summary
    shap_values, features = visualisations.get_shap_values(transformed_data, regressor2)

    st.subheader("SHAP Summary: Away Team")
    shap.initjs()
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, transformed_data[regressor2.feature_names_], show=True
    )
    plt.title("Prediction for Away Team Goals")
    plt.xlabel("Shap Impact")
    st.pyplot(plt)

    plt.title("Match Result Performance")
    fig = visualisations.plot_performance_metrics_result(model_performance)
    st.pyplot(fig)

    plt.title("Home and Away Goals Performance")
    fig = visualisations.plot_mean_absolute_error_home_away(model_performance)
    st.pyplot(fig)


# Embed the table in the right column
with col2:
    # Right panel for detailed information about selected fixture
    st.header("Fixture Details")

    # Dropdown to select a fixture
    selected_fixture = st.selectbox(
        "Select a fixture:", current_fixtures["Fixture"].tolist()
    )
    st.subheader(f"Selected Fixture: {selected_fixture}")

    # Badges
    s3_helpers.display_side_by_side_images(selected_fixture)

    # Current fixture
    current_prediction = visualisations.extract_current_result(
        current_fixtures, selected_fixture
    )
    st.dataframe(current_prediction)

    # Then display your plot
    fig, ax = visualisations.plot_last_5(transformed_data, selected_fixture)
    st.pyplot(fig)

    # Waterfall
    st.subheader("Home Team Goals")

    # Plot
    fig, ax = visualisations.create_waterfall(
        transformed_data, regressor1, selected_fixture
    )
    st.pyplot(fig)

    st.subheader("Away Team Goals")

    # Plot
    fig, ax = visualisations.create_waterfall(
        transformed_data, regressor2, selected_fixture
    )
    st.pyplot(fig)

    # Stadium
    st.subheader("Stadium Details")

    # Dropdown for selecting a stadium
    selected_stadium = current_fixtures[
        current_fixtures["Fixture"] == selected_fixture
    ]["Location"].values[0]

    # Get the selected stadium's information
    stadium_info = stadium_data[stadium_data["Stadium"] == selected_stadium]

    # Display stadium information
    st.write(f"Stadium: {stadium_info['Stadium'].values[0]}")
    st.write("Capacity:" + "{: ,}".format(stadium_info["Capacity"].values[0]))
    # Plot the selected stadium on a map
    reset_view = st.button("Show/Reset view")

    if reset_view:
        st.pydeck_chart(
            pdk.Deck(
                map_style="mapbox://styles/mapbox/satellite-streets-v11",
                initial_view_state=pdk.ViewState(
                    latitude=stadium_info["Latitude"].values[0],
                    longitude=stadium_info["Longitude"].values[0],
                    zoom=15,
                ),
            )
        )

# Embedding the JavaScript
keep_alive_script = """
<script>
function keepAlive(){
    fetch("/"); // Replace with a lightweight endpoint if necessary
    console.log("Keep-alive ping sent");
}

// Ping every 5 minutes (30000000 milliseconds)
setInterval(keepAlive, 30000000);
</script>
"""

st.markdown(keep_alive_script, unsafe_allow_html=True)
