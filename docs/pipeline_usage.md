# Explanation Pipeline

1. **Input selection**: User filters fuel type → Genmodel → Genmodel_ID.
2. **Tabular explanation**: SHAP is computed to show feature drivers (e.g., high mileage reduced value).
3. **Image explanation**: Grad-CAM highlights visual regions important to the prediction (e.g., headlights, front grille).
4. **Narration**: The agent combines SHAP (tabular) + Grad-CAM (image) + Knowledge Docs to generate a coherent story.
5. **Chat Layer**: Users can ask natural questions (e.g., "Why engine size matters?" or "What if year was newer?").
