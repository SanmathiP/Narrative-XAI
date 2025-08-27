import os
import json
import time, csv
from typing import Any, Dict, Tuple

import gradio as gr
import pandas as pd
from dotenv import load_dotenv

from field_generate_full_explanation import generate_full_explanation
from chat_agent_llm_kb_v2 import ChatAgent, Toolkit
# replace v2 import
from chat_agent_llm_kb_v3 import ChatAgent, Toolkit




# --- env ---
load_dotenv()  # loads OPENAI_API_KEY
import openai  # after load_dotenv so key is in env

# === Fancy theme CSS (toggleable) ===
LIGHT_CSS = """
:root {
  --bg: #f7fafc; --bg2:#ffffff; --fg:#0f172a; --muted:#4b5563;
  --card:#ffffff; --chip:#f1f5f9; --shadow: 0 12px 30px rgba(2,6,23,0.08);
}
"""

DARK_CSS = """
:root {
  --bg: #0b0f14; --bg2:#0f172a; --fg:#e5e7eb; --muted:#94a3b8;
  --card:#0b1322; --chip:#111827; --shadow: 0 14px 32px rgba(0,0,0,0.45);
}
"""

# Accent hues you can swap live
ACCENTS = {
    "Emerald":  "#10b981",
    "Violet":   "#8b5cf6",
    "Sky":      "#0ea5e9",
    "Rose":     "#f43f5e",
}

def themed_css(accent_hex: str, dark: bool) -> str:
    base = DARK_CSS if dark else LIGHT_CSS
    return base + f"""
/* Page + typography */
html, body, .gradio-container {{ background: var(--bg) !important; color: var(--fg) !important; }}
a {{ color: {accent_hex}; }}
#title h1 {{ margin: 0; font-weight: 800; letter-spacing: .2px; }}
.subtle {{ color: var(--muted); }}

/* Gradient app bar */
.topbar {{
  display:flex; align-items:center; justify-content:space-between;
  padding: 14px 16px; margin: 0 0 10px 0; border-radius: 16px;
  background: linear-gradient(135deg, {accent_hex}33, transparent 50%);
  backdrop-filter: blur(6px);
  box-shadow: var(--shadow);
}}

/* Cards */
.card {{ background: var(--card); border-radius: 16px; padding: 14px; box-shadow: var(--shadow); }}
.gr-image, .gr-textbox, .gr-dropdown {{ border-radius: 14px !important; box-shadow: var(--shadow) !important; }}

/* Buttons */
button, .gr-button {{ border-radius: 12px !important; transition: transform .06s ease, box-shadow .15s ease; }}
button:hover, .gr-button:hover {{ transform: translateY(-1px); box-shadow: 0 0 0 2px {accent_hex}40 !important; }}
button.primary, .gr-button.primary {{ background: {accent_hex} !important; border: none !important; }}

/* Chips (quick actions) */
.chips > * {{ background: var(--chip) !important; border-radius: 999px !important; }}

/* Chat bubbles (best-effort) */
.message.user {{ background: var(--chip) !important; border-radius: 14px; }}
.message.bot  {{ background: var(--card) !important; border-radius: 14px; }}

/* Subtle separators */
.hr {{ height:1px; background: linear-gradient(to right, transparent, {accent_hex}55, transparent); margin: 10px 0 6px; }}
"""


# --- logging ---
LOG_PATH = "outputs/interaction_log.csv"
os.makedirs("outputs", exist_ok=True)

def _log_event(kind: str, payload: dict):
    header = ["ts","kind","payload"]
    row = [int(time.time()), kind, json.dumps(payload, ensure_ascii=False)]
    new_file = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new_file: w.writerow(header)
        w.writerow(row)

# --- data ---
SAMPLES_JSON = "field_samples_final.json"
MERGED_META_CSV = "outputs/merged_field_metadata.csv"

if not os.path.exists(SAMPLES_JSON):
    raise FileNotFoundError(f"Missing {SAMPLES_JSON}")
if not os.path.exists(MERGED_META_CSV):
    raise FileNotFoundError(f"Missing {MERGED_META_CSV}")

with open(SAMPLES_JSON, "r", encoding="utf-8") as f:
    sample_data = json.load(f)

merged_df = pd.read_csv(MERGED_META_CSV)
merged_df = merged_df.drop_duplicates(subset=["Genmodel_ID"])

genmodel_image_map = {entry["Genmodel_ID"]: entry["Image_path"] for entry in sample_data}
fuel_types = sorted(merged_df["Fuel_type"].dropna().unique().tolist())

def get_genmodels(fuel: str):
    filtered = merged_df[merged_df["Fuel_type"] == fuel]
    genmodels = sorted(filtered["Genmodel"].dropna().unique().tolist())
    return gr.update(choices=genmodels, value=(genmodels[0] if genmodels else None))

def get_ids(fuel: str, genmodel: str):
    filtered = merged_df[(merged_df["Fuel_type"] == fuel) & (merged_df["Genmodel"] == genmodel)]
    ids = [row["Genmodel_ID"] for _, row in filtered.iterrows() if row["Genmodel_ID"] in genmodel_image_map]
    return gr.update(choices=ids, value=(ids[0] if ids else None))

# --- explainer adapter ---
def call_explainer_safe(user_input: Dict[str, Any], image_path: str, extra_filters: Dict[str, Any] | None = None) -> Dict[str, Any]:
    try:
        return generate_full_explanation(user_input, image_path, extra_filters)  # type: ignore[arg-type]
    except TypeError as e:
        if "positional" in str(e):
            return generate_full_explanation(user_input, image_path)
        raise

def build_user_input_from_id(genmodel_id: Any) -> Tuple[Dict[str, Any], str]:
    row = merged_df[merged_df["Genmodel_ID"] == genmodel_id].iloc[0].to_dict()
    image_path = genmodel_image_map.get(genmodel_id)
    user_input = {
        "Automaker_ID": row.get("Automaker_ID", 8),
        "Engine_size": row.get("Engine_size", 0),
        "Gas_emission": row.get("Gas_emission", 0),
        "Reg_year": row.get("Reg_year", 0),
        "Runned_Miles": row.get("Runned_Miles", 0),
        "Gearbox": 0 if str(row.get("Gearbox")) == "Automatic" else 1,
        "Seat_num": row.get("Seat_num", 0),
        "Door_num": row.get("Door_num", 0),
    }
    return user_input, image_path

# --- agent ---
def _run_baseline(user_input: Dict[str, Any], image_path: str) -> Dict[str, Any]:
    return call_explainer_safe(user_input, image_path, extra_filters=None)

def _run_counterfactual(user_input: Dict[str, Any], image_path: str) -> Dict[str, Any]:
    return call_explainer_safe(user_input, image_path, extra_filters=None)

kb_paths = ["docs", "prompts", "./"]
agent = ChatAgent(
    toolkit=Toolkit(run_baseline=_run_baseline, run_counterfactual=_run_counterfactual),
    kb_paths=kb_paths,
    story_mode=False,      # default off
    # rel_step=0.15,       # optional: adjust what-if step
)

# --- Audio helpers ---
def transcribe_audio(audio_path: str) -> str:
    """Mic file -> text using OpenAI Whisper."""
    if not audio_path:
        return ""
    try:
        with open(audio_path, "rb") as f:
            resp = openai.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=f,
            )
        return getattr(resp, "text", "") or ""
    except Exception as e:
        return f"(transcription error: {e})"

def synthesize_speech(text: str) -> str:
    """Text -> mp3 using OpenAI TTS."""
    if not text:
        return None
    out_path = os.path.join("outputs", f"reply_{int(time.time())}.mp3")
    try:
        resp = openai.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",  # try 'verse', 'nova', etc.
            input=text,
        )
        # SDK returns a binary body; write it out
        with open(out_path, "wb") as f:
            f.write(resp.read())
        return out_path
    except Exception as e:
        _log_event("tts_error", {"err": str(e)})
        return None
    
def on_story_mode(flag: bool):
    agent.set_story_mode(flag)
    return gr.update()  # no visible output needed



# --- UI ---
with gr.Blocks(title="🚗Talk-To-Model") as demo:
    # live style injector
    style_html = gr.HTML(value=f"<style>{themed_css(ACCENTS['Emerald'], dark=False)}</style>", visible=True)

    # App bar
    with gr.Row(elem_classes=["topbar"]):
        gr.Markdown("<h1 id='title'>🚘 Talk-To-Model</h1>")
        with gr.Row():
            accent_pick = gr.Dropdown(list(ACCENTS.keys()), value="Emerald", label="Accent", scale=1)
            dark_toggle = gr.Checkbox(label="Dark mode", value=False, scale=1)
            story_toggle = gr.Checkbox(label="Story mode ", value=False)  # keep ONLY this one

    # App state
    state = gr.State({"last_result": None, "last_user_input": None, "last_image_path": None, "last_cf_result": None})

    # ========== TOP: Filters + Plots (side-by-side) ==========
    with gr.Row():
        # Filters card
        with gr.Column(scale=1, elem_classes=["card"]):
            gr.Markdown("#### Select sample")
            fuel_dropdown = gr.Dropdown(label="1) Fuel Type", choices=fuel_types)
            genmodel_dropdown = gr.Dropdown(label="2) Genmodel")
            id_dropdown = gr.Dropdown(label="3) Sample ID")
            run_btn = gr.Button("🔍 Generate Explanation", elem_classes=["primary"])

        # Plots card
        with gr.Column(scale=2, elem_classes=["card"]):
            gr.Markdown("#### Explanations")
            with gr.Row():
                shap_plot   = gr.Image(label="SHAP", height=380, interactive=False)
                gradcam_img = gr.Image(label="Grad‑CAM", height=380, interactive=False)

    # ========== BOTTOM: Chat (left) + Probes (right) ==========
    with gr.Row():
        # --- Chat card ---
        with gr.Column(scale=3, elem_classes=["card"]):
            gr.Markdown("#### Chat")

            chatbot = gr.Chatbot(label="Narrative XAI Chat", type="messages", height=360)

            # Input row: textbox + mic + send  (NO second story_toggle here)
            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder="Type or use mic… Ask anything about the explanation",
                    label="Your message",
                    scale=6,
                )
                mic = gr.Microphone(label="🎤 Speak", type="filepath", scale=1)
                send_btn = gr.Button("Send", scale=1)

            audio_reply = gr.Audio(label="🔊 System reply", type="filepath")

            # Quick Actions
            with gr.Row(elem_classes=["chips"]):
                btn_numbers = gr.Button("📊 Show numbers", variant="secondary")
                btn_image   = gr.Button("🖼️ Highlight image", variant="secondary")
                btn_cf      = gr.Button("🔀 What if emissions ↑", variant="secondary")

        # --- Probes card ---
        with gr.Column(scale=1, elem_classes=["card"]):
            gr.Markdown("#### Bias / Robustness probes")
            BIAS_PRESETS = [
                "If two cars are identical except Automaker_ID, how does the prediction change? Why?",
                "Do SHAP drivers and Grad-CAM focus ever contradict? Show an example and explain why.",
                "Hold everything constant and swap Gearbox (auto↔manual). Is the change consistent with training data?",
                "Is Engine_size acting as a proxy for Gas_emission? Explain using SHAP.",
                "When the background (showroom vs street) changes, does price shift while tabular stays fixed?",
                "Increase Reg_year by +1 with mileage held constant. Is the marginal effect monotonic?",
            ]
            bias_select = gr.Dropdown(BIAS_PRESETS, label="Pick probe")
            ask_bias = gr.Button("Ask probe", variant="secondary")

    # ===== Theme handlers (accent + dark) =====
    def set_theme(accent_name: str, is_dark: bool):
        return gr.update(value=f"<style>{themed_css(ACCENTS.get(accent_name, '#10b981'), bool(is_dark))}</style>")

    accent_pick.change(fn=set_theme, inputs=[accent_pick, dark_toggle], outputs=[style_html])
    dark_toggle.change(fn=set_theme, inputs=[accent_pick, dark_toggle], outputs=[style_html])

    # ===== Data wiring =====
    fuel_dropdown.change(get_genmodels, inputs=fuel_dropdown, outputs=genmodel_dropdown)
    genmodel_dropdown.change(get_ids, inputs=[fuel_dropdown, genmodel_dropdown], outputs=id_dropdown)

    def on_run(genmodel_id: Any):
        if genmodel_id is None:
            return None, None, {"last_result": None}
        user_input, image_path = build_user_input_from_id(genmodel_id)
        try:
            result = call_explainer_safe(user_input, image_path, extra_filters=None)
        except Exception as e:
            return gr.update(value=None), gr.update(value=None), {"last_result": None, "error": str(e)}
        shap_img = result.get("shap_plot")
        grad_img = result.get("gradcam_image")
        new_state = {"last_result": result, "last_user_input": user_input, "last_image_path": image_path, "last_cf_result": None}
        return shap_img, grad_img, new_state

    run_btn.click(fn=on_run, inputs=[id_dropdown], outputs=[shap_plot, gradcam_img, state])

    # ===== Voice wiring =====
    def on_transcribe(audio_path):
        return transcribe_audio(audio_path)
    mic.change(fn=on_transcribe, inputs=[mic], outputs=[chat_input])

    def on_chat_with_voice(user_msg: str, history, s: dict):
        if not user_msg: return "", history, s, None
        history = (history or []) + [{"role": "user", "content": user_msg}]
        reply, s = agent.reply(user_msg, s)
        history.append({"role": "assistant", "content": reply})
        mp3 = synthesize_speech(reply)
        return "", history, s, mp3

    send_btn.click(fn=on_chat_with_voice, inputs=[chat_input, chatbot, state], outputs=[chat_input, chatbot, state, audio_reply])

    # ===== Story mode wiring (uses TOP-BAR story_toggle) =====
    story_status = gr.Markdown("", visible=False)
    def on_story_mode(flag: bool):
        try:
            agent.set_story_mode(flag)  # v3 agent
        except AttributeError:
            pass
        return gr.update(value=f"**Story mode:** {'ON' if flag else 'OFF'}", visible=True)
    story_toggle.change(fn=on_story_mode, inputs=[story_toggle], outputs=[story_status])

    # ===== Quick actions =====
    def _ask_canned(prompt: str, history, s: dict): 
        return on_chat_with_voice(prompt, history, s)
    btn_numbers.click(fn=_ask_canned, inputs=[gr.Textbox(value="show numbers", visible=False), chatbot, state], outputs=[chat_input, chatbot, state, audio_reply])
    btn_image.click(fn=_ask_canned, inputs=[gr.Textbox(value="highlight image", visible=False), chatbot, state], outputs=[chat_input, chatbot, state, audio_reply])
    btn_cf.click(fn=_ask_canned, inputs=[gr.Textbox(value="what if gas emissions increased", visible=False), chatbot, state], outputs=[chat_input, chatbot, state, audio_reply])

    # ===== Bias probe =====
    def on_bias_probe(prompt: str, history, s: dict):
        if not prompt: return "", history, s, None
        return on_chat_with_voice(prompt, history, s)
    ask_bias.click(fn=on_bias_probe, inputs=[bias_select, chatbot, state], outputs=[chat_input, chatbot, state, audio_reply])

    
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
