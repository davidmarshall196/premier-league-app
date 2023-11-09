
import inference_pipeline
import data_extraction
import visualisations
from tabulate import tabulate
import constants
import shap
import joblib
import matplotlib.pyplot as plt
import gradio as gr

# Grab data
transformed_data = inference_pipeline.run_inference_pipeline()




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

# Define your interfaces here

shap_summary_interface = gr.Interface(
    fn=visualisations.shap_summary,
    inputs=[],
    outputs=[gr.Plot()],
)
shap_summary_interface
with gr.Blocks() as demo:
    gr.Markdown("""
    Premier League Predictior. Predict the next 10 fixtures.
    """)
    with gr.Column():
        shap_summary_interface
        with gr.Row():
            print(tabulate(table))

        
shap_summary_interface
with gr.Blocks() as demo:
    gr.Markdown("""
    Premier League Predictior. Predict the next 10 fixtures.
    """)
    with gr.Row():
            gr.HTML(tabulate(table, 
                           tablefmt='fancy_grid', 
                           headers='keys'))


demo.launch()






