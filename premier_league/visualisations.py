import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from typing import Any, Tuple, List
from colorama import Fore, Style
from tabulate import tabulate
import shap
import joblib

try:
    from premier_league import constants
except ImportError:
    import constants


def get_shap_values(data: pd.DataFrame, classifier) -> Tuple[np.ndarray, List[str]]:
    """
    Compute SHAP values for the given data using the specified classifier.

    Parameters:
    - data (pd.DataFrame): The input data for which SHAP values are to be computed.
    - classifier: The trained classification model.

    Returns:
    - Tuple[np.ndarray, List[str]]: A tuple containing the SHAP values and the
    feature names.

    Notes:
    - Uses the SHAP library to explain the output of the model.
    - The SHAP values provide insight into the contribution of each feature to the
    model's predictions.
    """
    explainer = shap.Explainer(classifier)
    shap_values = explainer.shap_values(data[classifier.feature_names_])
    return shap_values, classifier.feature_names_


def shap_summary():
    """
    This function creates a SHAP summary plot for the given data and classifier.

    Parameters:
    - None

    Returns:
    - matplotlib.figure.Figure
    """
    # Grab data
    transformed_data = pd.read_csv("../data/transformed_data_predictions.csv")
    classifier = joblib.load(constants.CLASS_MODEL_NAME)

    # calculate SHAP values
    explainer = shap.Explainer(classifier)
    shap_values = explainer.shap_values(transformed_data[classifier.feature_names_])
    fig_m = plt.figure(tight_layout=True)
    shap.summary_plot(
        shap_values,
        transformed_data[classifier.feature_names_],
        plot_type="bar",
        show=False,
    )
    plt.title("Result Prediction Feature Importances")
    plt.xlabel("Shap Impact")
    return fig_m


def actuals_predicted(x_test: pd.DataFrame, ha: str = "Home") -> None:
    """
    Plot actual vs predicted values for Home or Away goals.

    Parameters:
    - x_test (pd.DataFrame): The test dataset containing actual and predicted values.
    - ha (str): Specifies 'Home' or 'Away'. Defaults to 'Home'.

    Notes:
    - Plots a scatterplot and regression line to compare actual and predicted goals.
    - Adds a slight noise to the data for better visualization.
    """
    plot_data = x_test.copy()
    plot_data[ha] = plot_data[f"{ha} Prediction"].astype(str)
    plot_data[f"{ha} Prediction Noise"] = plot_data[f"{ha} Prediction"].apply(
        lambda x: x - random.randint(-400, 400) / 1000
    )
    plot_data = plot_data.rename(
        columns={"FTHG": "Actual Home Goals", "FTAG": "Actual Away Goals"}
    )
    plot_data[f"Actual {ha} Goals Noise"] = plot_data[f"Actual {ha} Goals"].apply(
        lambda x: x - random.randint(-400, 400) / 1000
    )
    plot_data[ha] = plot_data[f"{ha} Prediction"].astype(int).astype(str)
    sns.scatterplot(
        data=plot_data, x=f"{ha} Prediction Noise", y=f"Actual {ha} Goals Noise", hue=ha
    )
    sns.regplot(
        data=plot_data,
        x=f"{ha} Prediction Noise",
        y=f"Actual {ha} Goals Noise",
        scatter=False,
    )
    plt.title(f"Actual v Predicted, {ha} Goals")
    plt.xlabel("Predicted Goals")
    plt.ylabel("Actual Goals")
    plt.show()


def histoplot(x_test: pd.DataFrame, ha: str = "Home") -> None:
    """
    Create a histogram plot for the difference between actual and predicted goals.

    Parameters:
    - x_test (pd.DataFrame): The test dataset containing actual and predicted values.
    - ha (str): Specifies 'Home' or 'Away'. Defaults to 'Home'.

    Notes:
    - Visualizes the distribution of the difference between actual and predicted goal
    counts.
    - Useful for assessing the accuracy of the predictions.
    """
    plot_data = x_test.copy()
    plot_data = plot_data.rename(
        columns={"FTHG": "Actual Home Goals", "FTAG": "Actual Away Goals"}
    )
    plot_data["Home Difference"] = (
        plot_data["Home Prediction"] - plot_data["Actual Home Goals"]
    )
    plot_data["Away Difference"] = (
        plot_data["Away Prediction"] - plot_data["Actual Away Goals"]
    )
    plot_data = plot_data.groupby(f"{ha} Difference").count().reset_index()
    plot_data = plot_data[[f"{ha} Difference", "season"]]
    plot_data.columns = [f"{ha} Difference", "Count"]
    sns.barplot(data=plot_data, x=f"{ha} Difference", y="Count", color="blue")
    plt.title(f"{ha} Goals Difference Count")
    plt.xlabel("Difference between actual and predicted")
    return


def plot_features(model, fa: str = "Home", n: int = 15) -> None:
    """
    Plot the top n feature importances of the given model.

    Parameters:
    - model: The machine learning model with feature importances.
    - fa (str): A label to include in the plot title, e.g., 'Home' or 'Away'. Defaults
    to 'Home'.
    - n (int): The number of top features to plot. Defaults to 15.

    Notes:
    - Scales the feature importances for better visualization.
    - Excludes 'match_prediction' from the feature importances.
    - Creates a bar plot showing the top n features by their importance.
    """
    scaler = MinMaxScaler((0, 100))
    fi = pd.DataFrame(
        {"Feature": model.feature_names_, "Raw Importance": model.feature_importances_}
    )
    fi["rank"] = fi["Raw Importance"].rank(ascending=False)
    fi = fi[fi["Feature"] != "match_prediction"]

    fi["Scaled Importance"] = scaler.fit_transform(
        fi["Raw Importance"].values.reshape(-1, 1)
    )
    plot_data = fi.sort_values(by="rank").head(n)
    sns.barplot(data=plot_data, x="Scaled Importance", y="Feature", color="Blue")
    plt.title(f"{fa} Model Feature Importance")
    plt.xlabel("Feature Importance")
    plt.show()


def extract_last_results(team: str, dataframe: pd.DataFrame) -> str:
    """
    Extracts the last 5 results for a given team from a DataFrame and returns them as
    a string.

    Parameters:
    - team (str): Name of the team.
    - dataframe (pd.DataFrame): DataFrame containing match data.

    Returns:
    - results (str): String representing the last 5 results for the given team.
    """
    sorted_dataframe = dataframe.copy()
    sorted_dataframe["SeasonStartYear"] = sorted_dataframe["season"].str[:4].astype(int)
    sorted_dataframe["SeasonEndYear"] = (
        sorted_dataframe["season"].str[:2] + sorted_dataframe["season"].str[5:]
    ).astype(int)
    sorted_dataframe["MonthNumber"] = pd.to_datetime(
        sorted_dataframe["MatchDayMonth"], format="%B"
    ).dt.month
    sorted_dataframe["FinalDate"] = np.where(
        (sorted_dataframe["MonthNumber"].astype(int) >= 8)
        & (sorted_dataframe["MonthNumber"].astype(int) <= 12),
        pd.to_datetime(
            sorted_dataframe["MatchDayMonth"].astype(str)
            + "/"
            + sorted_dataframe["MatchDayDate"].astype(str)
            + "/"
            + sorted_dataframe["SeasonStartYear"].astype(str),
            format="%B/%d/%Y",
            errors="coerce",
        ),
        pd.to_datetime(
            sorted_dataframe["MatchDayMonth"].astype(str)
            + "/"
            + sorted_dataframe["MatchDayDate"].astype(str)
            + "/"
            + sorted_dataframe["SeasonEndYear"].astype(str),
            format="%B/%d/%Y",
            errors="coerce",
        ),
    )
    sorted_dataframe = sorted_dataframe.sort_values(by="FinalDate", ascending=True)
    team_matches = sorted_dataframe[
        (
            (sorted_dataframe["HomeTeam"] == team)
            | (sorted_dataframe["AwayTeam"] == team)
        )
    ].tail(5)

    results = []
    counter = 0
    for _, row in team_matches.iterrows():
        if counter >= 5:
            break

        if row["HomeTeam"] == team:
            if row["FTR"] == 0:
                results.append([Fore.GREEN + "W"])
            elif row["FTR"] == 1:
                results.append([Fore.YELLOW + "D"])
            else:
                results.append([Fore.RED + "L"])
        else:
            if row["FTR"] == 0:
                results.append([Fore.RED + "L"])
            elif row["FTR"] == 1:
                results.append([Fore.YELLOW + "D"])
            else:
                results.append([Fore.GREEN + "W"])

        counter += 1

    # Reset color after printing
    results = [[Style.RESET_ALL + item[0]] for item in results]
    headers = ["Last 5 Games"]
    print(team)
    print(tabulate(results, headers=headers, tablefmt="fancy_grid"))
    return results


def format_results(
    matches: pd.DataFrame, home_team: str, away_team: str
) -> pd.DataFrame:
    """
    Format and return match results for the specified home and away teams.

    Parameters:
    - matches (pd.DataFrame): The DataFrame containing match data.
    - home_team (str): The name of the home team.
    - away_team (str): The name of the away team.

    Returns:
    - pd.DataFrame: A DataFrame containing formatted results for the home and away
    teams.

    Notes:
    - Assumes the DataFrame contains columns 'HomeTeam', 'AwayTeam', and 'FTR'.
    - 'FTR' is assumed to be coded as 0 for home win, 1 for draw, and 2 for away win.
    - Formats results as strings describing the outcome from the perspective of each
    team.
    """
    matches_df = matches.copy()

    # Define a function to convert the result code to a string
    def result_string(row, team):
        if row["HomeTeam"] == team:
            if row["FTR"] == 0:
                return "Win"
            elif row["FTR"] == 1:
                return "Draw"
            elif row["FTR"] == 2:
                return "Loss"
        else:
            if row["FTR"] == 0:
                return "Loss"
            elif row["FTR"] == 1:
                return "Draw"
            elif row["FTR"] == 2:
                return "Win"
        return "NA"

    # Process results for home team
    home_results = []
    for index, row in matches_df.iterrows():
        if row["HomeTeam"] == home_team or row["AwayTeam"] == home_team:
            opponent = (
                row["AwayTeam"] if row["HomeTeam"] == home_team else row["HomeTeam"]
            )
            venue = "H" if row["HomeTeam"] == home_team else "A"
            result = f"{result_string(row, home_team)} v {opponent} ({venue})"
            home_results.append(result)

    # Process results for away team
    away_results = []
    for index, row in matches_df.iterrows():
        if row["HomeTeam"] == away_team or row["AwayTeam"] == away_team:
            opponent = (
                row["AwayTeam"] if row["HomeTeam"] == away_team else row["HomeTeam"]
            )
            venue = "H" if row["HomeTeam"] == away_team else "A"
            result = f"{result_string(row, away_team)} v {opponent} ({venue})"
            away_results.append(result)

    # Create final DataFrame
    final_df = pd.DataFrame({home_team: home_results, away_team: away_results})

    return final_df


def extract_last_fixtures(fixture: str, dataframe: pd.DataFrame) -> str:
    """
    Extracts the last 5 results for a given team from a DataFrame and returns them
    as a string.

    Parameters:
    - fixture (str): Selected fixture.
    - dataframe (pd.DataFrame): DataFrame containing match data.

    Returns:
    - results (str): String representing the last 5 results for the given team.
    """
    sorted_dataframe = dataframe.copy()
    sorted_dataframe["SeasonStartYear"] = sorted_dataframe["season"].str[:4].astype(int)
    sorted_dataframe["SeasonEndYear"] = (
        sorted_dataframe["season"].str[:2] + sorted_dataframe["season"].str[5:]
    ).astype(int)
    sorted_dataframe["MonthNumber"] = pd.to_datetime(
        sorted_dataframe["MatchDayMonth"], format="%B"
    ).dt.month
    sorted_dataframe["FinalDate"] = np.where(
        (sorted_dataframe["MonthNumber"].astype(int) >= 8)
        & (sorted_dataframe["MonthNumber"].astype(int) <= 12),
        pd.to_datetime(
            sorted_dataframe["MatchDayMonth"].astype(str)
            + "/"
            + sorted_dataframe["MatchDayDate"].astype(str)
            + "/"
            + sorted_dataframe["SeasonStartYear"].astype(str),
            format="%B/%d/%Y",
            errors="coerce",
        ),
        pd.to_datetime(
            sorted_dataframe["MatchDayMonth"].astype(str)
            + "/"
            + sorted_dataframe["MatchDayDate"].astype(str)
            + "/"
            + sorted_dataframe["SeasonEndYear"].astype(str),
            format="%B/%d/%Y",
            errors="coerce",
        ),
    )
    sorted_dataframe = sorted_dataframe.sort_values(by="FinalDate", ascending=True)
    sorted_dataframe["fixture"] = (
        sorted_dataframe["HomeTeam"] + " v " + sorted_dataframe["AwayTeam"]
    )
    home_team = fixture.split(" v ")[0]
    away_team = fixture.split(" v ")[1]

    current_season = sorted(sorted_dataframe["season"].unique())[-1]
    selected_index = sorted_dataframe[
        (sorted_dataframe["fixture"] == fixture)
        & (sorted_dataframe["season"] == current_season)
    ].index[0]

    # Filter rows before the selected index
    filtered_df = sorted_dataframe.iloc[:selected_index]

    home_matches = filtered_df[
        (
            (filtered_df["HomeTeam"] == home_team)
            | (filtered_df["AwayTeam"] == home_team)
        )
    ].tail(5)[["HomeTeam", "AwayTeam", "FTR"]]
    away_matches = filtered_df[
        (
            (filtered_df["HomeTeam"] == away_team)
            | (filtered_df["AwayTeam"] == away_team)
        )
    ].tail(5)[["HomeTeam", "AwayTeam", "FTR"]]
    if home_matches.shape[0] == 0:
        home_matches["HomeTeam"] = ["Not Available"] * 5
        home_matches["AwayTeam"] = ["Not Available"] * 5
    if away_matches.shape[0] == 0:
        away_matches["HomeTeam"] = ["Not Available"] * 5
        away_matches["AwayTeam"] = ["Not Available"] * 5
    matches = pd.concat([home_matches, away_matches])
    return format_results(matches, home_team, away_team), home_team, away_team


def get_date_suffix(day: int) -> str:
    """
    Returns the suffix for a given date.

    Parameters:
    - day (int): The day of the month.

    Returns:
    - str: The date with the appropriate suffix.
    """
    if 4 <= day <= 20 or 24 <= day <= 30:
        suffix = "th"
    else:
        suffix = ["st", "nd", "rd"][day % 10 - 1]

    return str(day) + suffix


def grab_fixture_info(current_fixtures, selected_fixture):
    """
    Returns the suffix for a given date.

    Parameters:
    - current_fixtures (pd.DataFrame): The current fixtures.
    - selected_fixture (str): The fixture to dispaly

    Returns:
    - None
    """
    c_fix = current_fixtures[
        [
            "HomeTeam",
            "AwayTeam",
            "Home Prediction",
            "Away Prediction",
            "Date",
            "Fixture Time",
            "Location",
            "Fixture",
            "MatchDayDay",
            "MatchDayMonth",
            "MatchDayDate",
        ]
    ]
    c_fix = c_fix[c_fix["Fixture"] == selected_fixture]
    c_fix["Prediction"] = (
        c_fix["Home Prediction"].astype(str)
        + "-"
        + c_fix["Away Prediction"].astype(str)
    )
    c_fix["day_with_suffix"] = c_fix["MatchDayDate"].astype(int).apply(get_date_suffix)
    c_fix["Date"] = (
        c_fix["MatchDayDay"].astype(str)
        + " "
        + c_fix["day_with_suffix"].astype(str)
        + " "
        + c_fix["MatchDayMonth"].astype(str)
    )
    c_fix = c_fix[
        ["HomeTeam", "Prediction", "AwayTeam", "Date", "Fixture Time", "Location"]
    ].reset_index(drop=True)
    c_fix.columns = ["Home", "Prediction", "Away", "Date", "Time", "Stadium"]

    return c_fix


def tab_fixture_prediction(
    current_fixtures: pd.DataFrame, selected_fixture: str
) -> str:
    """
    Tabulate the fixture prediction for a selected fixture.

    Parameters:
    - current_fixtures (pd.DataFrame): DataFrame containing current fixtures.
    - selected_fixture (str): The identifier for the selected fixture.

    Returns:
    - str: A string representing the tabulated prediction of the selected fixture.

    Notes:
    - Utilizes the `grab_fixture_info` function to extract information about the
    selected fixture.
    - Focuses on the 'Home', 'Prediction', and 'Away' columns for display.
    - Returns the tabulated data in a visually appealing 'fancy_grid' format.
    """
    c_fix = grab_fixture_info(current_fixtures, selected_fixture)
    input_df = c_fix[["Home", "Prediction", "Away"]]
    return tabulate(input_df, tablefmt="fancy_grid", headers="keys")


def tab_fixture_details(current_fixtures: pd.DataFrame, selected_fixture: str) -> str:
    """
    Tabulate the fixture details for a selected fixture.

    Parameters:
    - current_fixtures (pd.DataFrame): DataFrame containing current fixtures.
    - selected_fixture (str): The identifier for the selected fixture.

    Returns:
    - str: A string representing the tabulated details of the selected fixture.

    Notes:
    - Utilizes the `grab_fixture_info` function to extract detailed information about
    the selected fixture.
    - Focuses on the 'Date', 'Time', and 'Stadium' columns for display.
    - Returns the tabulated data in a visually appealing 'fancy_grid' format.
    """
    c_fix = grab_fixture_info(current_fixtures, selected_fixture)
    input_df = c_fix[["Date", "Time", "Stadium"]]
    return tabulate(input_df, tablefmt="fancy_grid", headers="keys")


def pad_column(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Pad a specified column in the DataFrame so all strings have the same length.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the column to be padded.
    - column (str): The name of the column to be padded.

    Returns:
    - pd.DataFrame: The DataFrame with the specified column padded.

    Notes:
    - Pads strings in the specified column to the length of the longest string in
    that column.
    - The padding is applied on the left side of the strings.
    """
    # Determine the maximum string length
    max_length = data[column].str.len().max()

    # Pad each string to the maximum length
    data[column] = data[column].str.pad(width=int(max_length), side="left")
    # data[column] = '.' + data[column]
    return data


def extract_current_result(
    current_fixtures: pd.DataFrame, selected_fixture: str
) -> pd.DataFrame:
    """
    Extract the current result prediction for a selected fixture.

    Parameters:
    - current_fixtures (pd.DataFrame): DataFrame containing current fixtures and
    their predictions.
    - selected_fixture (str): The identifier for the selected fixture.

    Returns:
    - pd.DataFrame: A DataFrame with the current result prediction for the selected
    fixture.

    Notes:
    - Extracts home and away team predictions and combines them into a single
    DataFrame.
    - Formats the result for easy interpretation.
    """
    current_prediction = current_fixtures[
        current_fixtures["Fixture"] == selected_fixture
    ][["HomeTeam", "Home Prediction", "Away Prediction", "AwayTeam"]].reset_index(
        drop=True
    )
    current_prediction.columns = ["Home", "H", "A", "Away"]

    current_prediction_1 = current_prediction[["Home", "H"]]
    current_prediction_1.columns = ["Team", "Prediction"]
    current_prediction_2 = current_prediction[["Away", "A"]]
    current_prediction_2.columns = ["Team", "Prediction"]
    current_prediction = pd.concat(
        [current_prediction_1, current_prediction_2]
    ).reset_index(drop=True)
    current_prediction["Prediction"] = (
        current_prediction["Prediction"].astype(int).astype(str)
    )
    return current_prediction


def plot_last_5(df: pd.DataFrame, fixture: str) -> tuple:
    """
    Plot the last 5 matches for the teams in a given fixture.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the match data.
    - fixture (str): The fixture for which the last 5 matches will be plotted.

    Returns:
    - tuple: A tuple containing the figure and axis objects from matplotlib.

    Notes:
    - Uses the `extract_last_fixtures` function to obtain the last matches for
    the teams in the fixture.
    - Visualizes the outcomes (Win, Draw, Loss) using colored horizontal bars.
    - The results for the home team and away team are shown on the left and right
    sides of a central line, respectively.
    """
    format_res, home_team, away_team = extract_last_fixtures(fixture, df)

    format_res["col1"] = -1
    format_res["col2"] = 1

    results_to_colors = {"Loss": "red", "Draw": "orange", "Win": "green"}

    # Extract the results
    format_res["colour_1"] = (
        format_res[home_team].str.split().str[0].map(results_to_colors)
    )
    format_res["colour_2"] = (
        format_res[away_team].str.split().str[0].map(results_to_colors)
    )
    format_res = format_res.fillna("gray")
    # make plot
    df = format_res.copy()

    # Generate y-values for teams
    y_values = list(range(len(df)))

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(8, 3))

    # Plot results for teams
    ax.barh(y_values, df["col1"], color=df["colour_1"], alpha=0.6)
    ax.barh(y_values, df["col2"], color=df["colour_2"], alpha=0.6)

    # Add result text in the bars
    for i in range(len(df)):
        ax.text(df["col1"][i] / 2, i, df[home_team][i], ha="center", va="center")
        ax.text(df["col2"][i] / 2, i, df[away_team][i], ha="center", va="center")

    # Set labels and title
    ax.set_title("Last 5 Matches")

    # Hide axes
    ax.axis("off")

    # Add a vertical line in the middle
    ax.axvline(x=0, color="black", linestyle="--")

    # Add team names at the top of the plot
    ax.text(
        -0.5,
        len(df) - 0.5,
        home_team,
        ha="center",
        va="center",
        weight="bold",
        color="black",
        fontsize=14,
    )
    ax.text(
        0.5,
        len(df) - 0.5,
        away_team,
        ha="center",
        va="center",
        weight="bold",
        color="black",
        fontsize=14,
    )

    return fig, ax


def create_waterfall(
    transformed_data: pd.DataFrame, regression_model, fixture: str
) -> tuple:
    """
    Create a SHAP waterfall plot for a given fixture using the provided regression
    model.

    Parameters:
    - transformed_data (pd.DataFrame): The transformed data used by the regression
    model.
    - regression_model: The regression model used for prediction.
    - fixture (str): The fixture for which the SHAP waterfall plot will be created.

    Returns:
    - tuple: A tuple containing the figure and axis objects from matplotlib.

    Notes:
    - The SHAP waterfall plot shows the contribution of each feature to the model's
    prediction for a specific fixture.
    - Highlights how each feature affects the final prediction outcome.
    """
    transformed_data["fixture"] = (
        transformed_data["HomeTeam"].astype(str)
        + " v "
        + transformed_data["AwayTeam"].astype(str)
    )

    season = transformed_data["season"].unique()[-1]
    row = transformed_data[
        (transformed_data["season"] == season)
        & (transformed_data["fixture"] == fixture)
    ].index[0]

    # compute SHAP values
    explainer = shap.Explainer(regression_model)
    shap_values = explainer(transformed_data[regression_model.feature_names_])

    fig, ax = plt.subplots()
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values.values[row],
            base_values=explainer.expected_value,
            data=transformed_data[regression_model.feature_names_].iloc[row],
            feature_names=transformed_data[
                regression_model.feature_names_
            ].columns.tolist(),
        )
    )

    return fig, ax


def plot_mean_absolute_error_home_away(df: pd.DataFrame) -> Any:
    """
    Plots the Mean Absolute Error over time for 'home' and 'away' models from a given
    DataFrame.

    The function filters for 'home' and 'away' models, groups the data by run date
    and model type, and selects the model with the lowest mean absolute error for
    each date. It then plots these values over time for both model types.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the model data. Must include columns
    'Model type', 'Run Date', and 'mean_ae'.

    Returns:
    - Matplotlib plot object
    """
    df["Run Date"] = pd.to_datetime(df["Run Date"], format="%Y%m%d")
    # Filter for 'home' and 'away' models only
    df_filtered = df[df["Model type"].isin(["home", "away"])]

    # Group by Run Date and Model type, and take the model with the lowest mean_ae
    df_best = (
        df_filtered.groupby(["Run Date", "Model type"])
        .apply(lambda x: x.nsmallest(1, "mean_ae"))
        .reset_index(drop=True)
    )

    # Pivot the data for plotting
    df_pivot = df_best.pivot(index="Run Date", columns="Model type", values="mean_ae")

    # Creating the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_pivot.index, df_pivot["home"], label="Home Model", marker="o")
    ax.plot(df_pivot.index, df_pivot["away"], label="Away Model", marker="x")
    ax.set_title("Mean Absolute Error Over Time for Home and Away Models")
    ax.set_xlabel("Run Date")
    ax.set_ylabel("Mean Absolute Error")
    ax.legend()
    ax.grid(True)

    return fig


def plot_performance_metrics_result(df: pd.DataFrame) -> Any:
    """
    Plots the MCC, Accuracy, and F1 Score over time for 'result' models from a
    given DataFrame.

    The function filters for 'result' models, groups the data by run date, and
    selects the model with the lowest mean absolute error for each date. It then
    plots the MCC, Accuracy, and F1 Score values over time.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the model data. Must include columns
    'Model type', 'Run Date', 'mcc', 'accuracy', and 'f1'.

    Returns:
    - Matplotlib plot object
    """
    df["Run Date"] = pd.to_datetime(df["Run Date"], format="%Y%m%d")
    # Filter the data for 'result' models only
    df_result = df[df["Model type"] == "result"]

    # Group by Run Date and take the model with the best (lowest) mean_ae
    df_best_result = (
        df_result.groupby("Run Date")
        .apply(lambda x: x.nsmallest(1, "mean_ae"))
        .reset_index(drop=True)
    )

    # Creating the figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_best_result["Run Date"], df_best_result["mcc"], label="MCC", marker="o")
    ax.plot(
        df_best_result["Run Date"],
        df_best_result["accuracy"],
        label="Accuracy",
        marker="x",
    )
    ax.plot(
        df_best_result["Run Date"], df_best_result["f1"], label="F1 Score", marker="^"
    )
    ax.set_title("MCC, Accuracy, and F1 Score Over Time for Result Models")
    ax.set_xlabel("Run Date")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True)

    return fig
