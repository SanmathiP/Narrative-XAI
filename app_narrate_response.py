import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib

SHAP_PATH = "outputs/shap_input.csv"
MODEL_PATH = "outputs/price_model.pkl"
GRADCAM_PATH = "outputs/image_gradcam_overlay.jpg"
SHAP_PLOT_PATH = "outputs/shap_force_plot.png"
#works fine the commented def funtion
def generate_narrative():
    # Load data and model
    X = pd.read_csv(SHAP_PATH)
    model = joblib.load(MODEL_PATH)
    
    # Use first sample
    x_sample = X.iloc[[0]]
    
    # SHAP explanation
    explainer = shap.Explainer(model, X)
    shap_values = explainer(x_sample)

    # Convert SHAP values to simple text summary
    top_features = shap_values[0].values
    feature_names = x_sample.columns
    feature_impacts = sorted(
        zip(feature_names, top_features),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:3]  # top 3 impactful

    feature_text = ", ".join(
        f"{name} ({'+' if val > 0 else ''}{val:.1f})"
        for name, val in feature_impacts
    )

    # Build narrative
    narrative = {
        "title": "Multimodal Explanation for Car Pricing",
        "shap_plot": SHAP_PLOT_PATH,
        "gradcam_image": GRADCAM_PATH,
        "text_narrative": (
            f"The model estimated the car's price by considering key factors such as {feature_text}. "
            f"Visually, the GradCAM heatmap suggests that the model paid attention to specific parts of the car image, "
            f"likely related to body design or exterior condition."
        ),
        "non_tech_summary": (
            "Imagine explaining the car’s value like a mechanic checking both the logbook and the car’s photo. "
            "The model saw the year, mileage, and fuel type in the data, but also looked closely at visible features like paint or shape."
        ),
        "follow_up_prompt": "Would you like to ask a follow-up question about the explanation?"
    }

    return narrative

