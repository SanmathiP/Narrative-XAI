
# 🚘 Talk-To-Model — A Narrative XAI Framework for optimized cognitive effort conversations

This repository contains a research prototype developed for a master’s thesis on explainable AI (XAI) for augmented decision-making. The system combines tabular explanations (SHAP) and visual explanations (Grad-CAM) and narrates them in clear business language through a conversational interface. The core research goal is to reduce cognitive effort when interpreting model predictions by transforming raw explanations into concise, coherent narratives.

## Contents

- Project motivation and contributions

- System overview and architecture

- Installation and setup

- Data requirements and configuration

- Running the application

- Usage guide

- Narrative and explanation design

- Bias and robustness probes

- Voice input and output

- Limitations and future work

- Troubleshooting

- Licensing and citation

## 1. Motivation and Contributions

Modern predictive systems often output opaque predictions and equally opaque “explanations” (e.g., lists of feature importances). Users—particularly business stakeholders—struggle to connect these artifacts to actionable understanding. Two challenges recur:

1. Explanations are fragmented across modalities (e.g., a SHAP bar chart for tabular signals and a Grad-CAM overlay for image signals) without a unifying narrative.

2. Cognitive effort remains high: users must assemble the story themselves, infer implications, and decide what to do next.

This prototype contributes:

- A multimodal explanation pipeline that orchestrates SHAP (tabular) and Grad-CAM (image) into a single, coherent narrative.

- A conversational agent that answers natural language questions and supports “what-if” counterfactuals.

- Bias and robustness probes that guide users to evaluate fairness and stability.

- Optional voice input and text-to-speech output for accessibility and faster interaction.

## 2. System Overview and Architecture

The system is designed as an interactive prototype that allows users to select a vehicle sample, explore explanations of the model’s price prediction, and engage with these explanations through narrative and conversational interfaces. A typical interaction begins with the user selecting a sample by filtering on attributes such as fuel type, genmodel, and ID, followed by choosing an associated image. Once the sample is selected, the system generates explanations through two complementary methods. First, SHAP is applied to the tabular input features, producing feature-level attributions that indicate how each characteristic (for example, engine size, mileage, or registration year) influences the predicted price. Second, Grad-CAM is used to analyze the corresponding image, overlaying a heatmap that highlights which visual regions most influenced the model’s internal reasoning.

These two explanation modalities are then orchestrated together. Instead of presenting the user with separate plots, the system integrates their outputs into a coherent narrative. The narrative engine organizes the explanation into a structured story consisting of a key takeaway, the primary tabular drivers, the relevant visual evidence, the business implication of these factors, and a suggested action or counterfactual. This narrative is intended to reduce the cognitive burden on users by aligning the technical explanation with natural storytelling patterns.

The conversational layer sits on top of this pipeline. A chat agent manages user interaction, handling natural language questions such as “Why does engine size matter here?” or “What if emissions increased?” Depending on the query, the agent either invokes a counterfactual explanation, retrieves content from a local knowledge base, or prompts a language model to generate a contextualized narrative response. Users can switch between technical and business-oriented storytelling modes through a toggle. For accessibility, the system also supports voice input via speech-to-text and can produce spoken responses through text-to-speech.

Together, these components form an architecture that moves beyond raw visualizations toward a more user-friendly, multimodal explanation environment. By orchestrating SHAP and Grad-CAM into a conversational narrative, the system seeks to make complex AI predictions more comprehensible, actionable, and cognitively manageable for decision makers.


## 3. Installation and Setup

### 3.1 Clone the repository
```bash
git clone https://github.com/<your-username>/talk-to-model.git
cd talk-to-model
```

### 3.2 Create and activate a virtual environment
```bash
python -m venv .venv
```
  ### Windows
```bash
.venv\Scripts\activate
```

  ### macOS / Linux
```bash
source .venv/bin/activate
```

 ### 3.3 Install dependencies
```bash
pip install -r requirements.txt
```

The requirements include: Gradio (primary UI), SHAP (tabular explanations), PyTorch + Torchvision (Grad-CAM baseline), OpenAI (LLM, STT, TTS), pandas, matplotlib, scikit-learn, Pillow, OpenCV, python-dotenv.

 ### 3.4 Configure environment variables

Copy the template and set your key:

```bash
cp .env.example .env
```


Edit .env:
```bash
OPENAI_API_KEY=sk-xxxx...
```

The application reads OPENAI_API_KEY using python-dotenv.


## 4. Data Requirements and Configuration

This repository is configured for a small demonstration subset. It does not include the full DVM-CAR dataset.

Required local files (small, demo scale):

#### data/field_samples_final.json
JSON mapping Genmodel_ID to Image_path.

#### outputs/merged_field_metadata.csv
Metadata for dropdown filters (Fuel_type, Genmodel, Genmodel_ID).

#### outputs/shap_input.csv
Background data for SHAP explainer.

#### outputs/price_model.pkl
Trained regression model.

Do not commit large datasets. See docs/README_dataset.md for details.


## 5. Running the Application

From the repository root:
```bash
python apps/field_app_advanced_llm2.py
```

Open http://127.0.0.1:7860 in a browser.
The Streamlit version is in apps/app_streamlit.py.


## 6. Usage Guide

1. Select a fuel type, genmodel, sample ID, and image.

2.Click Generate Explanation.

3. View the SHAP force plot and Grad-CAM overlay.

4. Ask questions in the chat, e.g., “Why does engine size matter?”

5. Toggle Story mode to switch explanation style.

6. Use bias probes for robustness checks.

7. Optionally use the microphone to ask questions; listen to answers via TTS.



## 7. Narrative and Explanation Design
### 7.1 SHAP (Tabular)

SHAP values quantify the marginal contribution of each feature to the model’s prediction for the selected sample. The force plot shows direction and magnitude relative to a baseline.

### 7.2 Grad-CAM (Image)

Grad-CAM highlights which regions of the input image most influenced the model’s reasoning. This prototype uses MobileNet-V2 for demonstration.

### 7.3 Orchestration and Narration

Instead of showing two disconnected explanations, the system integrates them. The narrative engine generates a short story:

- Hook (main takeaway)

- Drivers (tabular feature importance)

- Visual (Grad-CAM evidence)

- Implication (business consequence)

- Action (next step or counterfactual)

## 8. Bias and Robustness Probes

The app includes preset questions for critical evaluation, such as:

- Are predictions consistent when changing registration year by one while holding mileage constant?

- Is engine size acting as a proxy for gas emission?

- Do SHAP and Grad-CAM ever contradict?

- Are predictions stable when gearbox type is swapped?

## 9. Voice Input and Output

- Speech-to-text: microphone input transcribed with Whisper or equivalent.

- Text-to-speech: system replies narrated back to the user.

- Both are optional.

## 10. Limitations and Future Work

### Limitations:

- Grad-CAM is coarse and qualitative.

- SHAP results depend on background data choice.

- Narratives depend on the language model prompt quality.

- Visual explanations use a baseline MobileNet, not a domain-specific model.

### Future work:

- Explore multimodal counterfactuals (varying tabular and image jointly).

- Add calibrated uncertainty reporting.

- Conduct structured human-subject evaluations.

- Improve fairness monitoring with automated probes.

### 11. Troubleshooting

- Styles not applied: clear browser cache, confirm CSS injection.

- SHAP/Grad-CAM blank: check file paths and confirm model compatibility.

- API errors: verify OPENAI_API_KEY is set in .env.

- CUDA issues: ensure correct torch/torchvision version or run CPU-only.

### 12. Licensing and Citation

- Choose a license (MIT recommended).

- Dataset attribution: [DVM-CAR dataset](https://deepvisualmarketing.github.io/)

- Cite SHAP, Grad-CAM, and this repository if used in academic work.
