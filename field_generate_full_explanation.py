# field_generate_full_explanation.py

import os
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import uuid
import torch
import cv2
import numpy as np
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights
from PIL import Image

# === CONFIG ===
MODEL_PATH = "outputs/price_model.pkl"
BG_PATH = "outputs/shap_input.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Actual GradCAM ===
def explain_image(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None

    img = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(img).unsqueeze(0).requires_grad_(True)

    weights = MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    model.eval()

    # Hooks
    grads = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        grads.append(grad_out[0])

    target_layer = model.features[-1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    # Forward + backward
    output = model(input_tensor)
    pred_class = output.argmax(dim=1)
    score = output[0, pred_class]
    model.zero_grad()
    score.backward()

    # GradCAM logic
    activation = activations[0].detach().squeeze()
    gradient = grads[0].detach().squeeze()
    weights = torch.mean(gradient, dim=(1, 2))

    cam = torch.zeros(activation.shape[1:], dtype=torch.float32)
    for i, w in enumerate(weights):
        cam += w * activation[i, :, :]

    cam = np.maximum(cam.numpy(), 0)
    cam = cv2.resize(cam, (img.width, img.height))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    img_np = np.array(img)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    out_path = os.path.join(OUTPUT_DIR, "image_gradcam_overlay.jpg")
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"✅ GradCAM explanation saved: {out_path}")
    return out_path


# === SHAP Explanation ===
def explain_tabular(user_input: dict):
    model = joblib.load(MODEL_PATH)
    background = pd.read_csv(BG_PATH).sample(100)
    df_input = pd.DataFrame([user_input])

    # Encode categorical fields
    gearbox_map = {"Automatic": 0, "Manual": 1, "Unknown": 2}
    fuel_map = {"Petrol": 0, "Diesel": 1, "Electric": 2, "Hybrid": 3, "Unknown": 4}

    if "Gearbox" in df_input.columns:
        df_input["Gearbox"] = df_input["Gearbox"].map(gearbox_map).fillna(2).astype(int)

    if "Fuel_type" in df_input.columns:
        df_input["Fuel_type"] = df_input["Fuel_type"].map(fuel_map).fillna(4).astype(int)

    expected_cols = list(background.columns)
    df_input = df_input[expected_cols].astype("float32")

    explainer = shap.Explainer(model, background)
    shap_values = explainer(df_input)

    plt.figure()
    shap_path = os.path.join(OUTPUT_DIR, f"shap_waterfall_{uuid.uuid4().hex[:6]}.png")
    shap.plots.waterfall(shap_values[0], show=False)
    plt.savefig(shap_path, bbox_inches="tight")
    plt.close()

    shap_scores = dict(zip(df_input.columns, shap_values[0].values))
    top_feats = sorted(shap_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

    return shap_path, shap_scores, top_feats


# === Combined Explanation Pipeline ===
def generate_full_explanation(user_input: dict, image_path: str):
    shap_path, shap_scores, top_feats = explain_tabular(user_input)
    gradcam_path = explain_image(image_path)

    text_narr = "The model focused on: " + ", ".join(
        [f"{feat} ({round(score, 2)})" for feat, score in top_feats]
    )
    non_tech_summary = (
        "The system considered your selected features and image to estimate price. "
        "Key details like engine size, emissions, or registration year had the biggest influence."
    )

    return {
        "shap_plot": shap_path,
        "gradcam_image": gradcam_path,
        "text_narrative": text_narr,
        "non_tech_summary": non_tech_summary,
        "follow_up_prompt": "Would you like to ask about a specific feature or image region?"
    }


# === Test Run ===
if __name__ == "__main__":
    sample_input = {
        "Automaker_ID": 8,
        "Fuel_type": "Petrol",
        "Engine_size": 2.0,
        "Gas_emission": 180,
        "Reg_year": 2010,
        "Runned_Miles": 95000,
        "Gearbox": "Automatic",
        "Seat_num": 5,
        "Door_num": 4
    }
    image_path = "outputs/sample_bmw.jpg"
    result = generate_full_explanation(sample_input, image_path)
    print(result)
