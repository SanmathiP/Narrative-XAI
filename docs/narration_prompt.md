# Narrative Prompt Templates for Explanations

## Structure
Every explanation should follow this structure:
1. **Gist** – A one-sentence summary of the main takeaway.  
2. **Bridge** – A short, clear link between tabular drivers (SHAP) and visual focus (Grad-CAM).  
3. **Next Action** – A gentle suggestion for what the user can do next (e.g., ask a what-if, request numbers, highlight image).

---

## Templates

### Template 1 – Feature Emphasis
**Gist:**  
“The model predicts a lower car price mainly due to the high mileage.”  

**Bridge:**  
“SHAP shows mileage as the strongest negative driver, while Grad-CAM highlights the worn look of the car body in the image, reinforcing this effect.”  

**Next Action:**  
“Would you like me to show how the prediction changes if mileage was reduced?”

---

### Template 2 – Positive Influence
**Gist:**  
“The newer registration year pushes the car value higher.”  

**Bridge:**  
“SHAP indicates year as a top positive factor, and Grad-CAM highlights modern design cues like the headlights.”  

**Next Action:**  
“Want me to drill down into the exact SHAP contribution or run a what-if for an older model?”

---

### Template 3 – Trade-off
**Gist:**  
“Engine size adds value but higher emissions reduce it.”  

**Bridge:**  
“SHAP shows both factors acting in opposite directions, while Grad-CAM highlights the engine bay and exhaust area in the image.”  

**Next Action:**  
“Shall I simulate what happens if the engine size is smaller?”

---

### Template 4 – Visual Emphasis
**Gist:**  
“The visual styling cues suggest a higher trim level.”  

**Bridge:**  
“Grad-CAM focuses on the grille and logo area, indicating luxury design, while SHAP attributes like seat count and door count provide supporting evidence.”  

**Next Action:**  
“Would you like me to explain how trim levels are encoded in the dataset?”

---

## Style Guidelines
- Use **simple, short sentences** (avoid jargon).  
- Always tie **SHAP drivers** and **Grad-CAM highlights** together.  
- End with an **invitation to continue the conversation** (to reduce cognitive dead-ends).  
- Never invent numbers — only report values from context.  
