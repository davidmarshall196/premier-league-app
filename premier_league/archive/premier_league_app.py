
import inference_pipeline
import data_extraction
import visualisations
from tabulate import tabulate
import constants
import shap
import joblib
import matplotlib.pyplot as plt
import gradio as gr
import pandas as pd

# Grab data
transformed_data = pd.read_csv(
    '../data/transformed_data_predictions.csv'
)

# Extract current fixtures
current_fixtures = data_extraction.extract_current_fixtures(
    transformed_data
)

# Extract results
results = visualisations.extract_last_results(
    'Brighton', 
    transformed_data
)

# Extract next fixtures
table = current_fixtures[['HomeTeam', 'AwayTeam', 
                  'Home Prediction', 'Away Prediction']].reset_index(drop=True)
table.index = table.index + 1
print(tabulate(table, tablefmt='fancy_grid', headers='keys'))


selected_fixture = "Bournemouth v West Ham"

# Load model
classifier = joblib.load(constants.class_model_name)

# Last 5 results
results = visualisations.extract_last_results( 'Arsenal', transformed_data)

import importlib
importlib.reload(visualisations)

# Unique fixtures
unique_fixtures = list(current_fixtures['Fixture'])


visualisations.shap_summary()

shap_summary_interface = gr.Interface(
    fn=shap_summary,
    inputs=[],
    outputs=gr.outputs.Image(type='pil'),
    description="""
    Premier League Predictor. Predict the next 10 fixtures.
    """
)

shap_summary_interface.launch()


import io
from PIL import Image
def shap_summary():
    transformed_data = pd.read_csv(
        '../data/transformed_data_predictions.csv'
    )
    classifier = joblib.load(constants.class_model_name)
    
    # calculate SHAP values
    explainer = shap.Explainer(classifier)
    shap_values = explainer.shap_values(
         transformed_data[classifier.feature_names_]
    )
    fig_m = plt.figure(tight_layout=True)
    shap.summary_plot(
        shap_values, 
        transformed_data[classifier.feature_names_],
        plot_type="bar",
        show=False
    )
    plt.title('Result Prediction Feature Importances')
    plt.xlabel('Shap Impact')
    # save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    # Create a PIL image
    img = Image.open(buf)
    return img


def wrapper_function():
    img = shap_summary()
    return img


iface = gr.Interface(fn=wrapper_function, inputs=None, outputs='image')
iface.launch()



