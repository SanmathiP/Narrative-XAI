# chat_agent_llm_kb_v2.py  (paste this whole file to replace your current v2)
from __future__ import annotations
import os, re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable, Tuple, List

try:
    import openai
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

from knowledge_base import KnowledgeBase

# ---------- parsing helpers ----------
_FEATURE_ALIASES = {
    "year": "Reg_year", "reg_year": "Reg_year",
    "mileage": "Runned_Miles", "miles": "Runned_Miles",
    "engine": "Engine_size", "engine_size": "Engine_size",
    "emission": "Gas_emission", "emissions": "Gas_emission",
    "gas emission": "Gas_emission", "gas emissions": "Gas_emission",
    "co2": "Gas_emission",
    "gearbox": "Gearbox",
}
_NUMERIC = {"Reg_year", "Runned_Miles", "Engine_size", "Gas_emission"}

def _cast_value(key: str, raw: str):
    raw = raw.strip().lower()
    if key in _NUMERIC:
        try:
            return float(raw) if "." in raw else int(float(raw))
        except Exception:
            return None
    if key == "Gearbox":
        if raw in {"auto","automatic","0"}: return 0
        if raw in {"manual","1"}: return 1
        return None
    return raw

def _find_feature_in_text(msg: str) -> Optional[str]:
    for alias, key in _FEATURE_ALIASES.items():
        if alias in msg:
            return key
    return None

def parse_what_if(message: str) -> Optional[Dict[str, Any]]:
    msg = message.lower()

    # Absolute edits: "what if X is 2012", "what if engine size = 1.6"
    m = re.search(r"what\s*-?\s*if\s+([a-z_ ]+)\s*(?:is|=|to)\s*([-+\w\.]+)", msg)
    if m:
        feat_raw, val_raw = m.group(1).strip(), m.group(2).strip()
        feat_key = _FEATURE_ALIASES.get(feat_raw.replace(" ","_")) or _FEATURE_ALIASES.get(feat_raw) or feat_raw
        val = _cast_value(feat_key, val_raw)
        return {feat_key: val} if val is not None else None

    # Relative edits: "what if X increased/decreased", "what if gas emissions were higher/lower"
    if any(w in msg for w in ["increase", "increased", "decrease", "decreased", "higher", "lower", "bigger", "larger", "smaller", "older", "newer"]):
        key = _find_feature_in_text(msg)
        if not key:
            return None
        # encode as relative directive; we will map to ±20% later (or ±1 year for Reg_year)
        if any(w in msg for w in ["decrease","decreased","smaller","lower","older"]):
            return {key: ("decrease","relative")}
        else:
            return {key: ("increase","relative")}

    return None

# ---------- toolkit ----------
@dataclass
class Toolkit:
    run_baseline: Callable[[Dict[str, Any], str], Dict[str, Any]]
    run_counterfactual: Callable[[Dict[str, Any], str], Dict[str, Any]]

class ChatAgent:
    def __init__(self, toolkit: Toolkit, model="gpt-4o-mini", temperature=0.2, kb_paths: List[str]|None=None,  rel_step: float = 0.2):
        self.toolkit = toolkit
        self.model = model
        self.temperature = temperature
        self.kb = KnowledgeBase(paths=kb_paths or [])
        self.rel_step = rel_step
        self.system_prompt = (
            "You are a Narrative XAI Agent. Reply in Gist → Bridge → Next Action. "
            "Tie SHAP (tabular) to Grad-CAM (visual). Use knowledge docs if helpful. "
            "Never invent numbers."
        )

    def _llm_available(self): 
        return _HAS_OPENAI and bool(os.environ.get("OPENAI_API_KEY"))

    def _render_context(self, state: Dict[str,Any]) -> str:
        lr = state.get("last_result") or {}
        ctx = []
        if lr.get("non_tech_summary"): ctx.append("summary:"+lr["non_tech_summary"][:160])
        if lr.get("text_narrative"): ctx.append("narrative:"+lr["text_narrative"][:160])
        if lr.get("top_features"): ctx.append("top_features:"+",".join([f["name"] for f in lr["top_features"][:5] if isinstance(f,dict) and f.get("name")]))
        if lr.get("image_regions"): ctx.append("image_focus:"+",".join([r["name"] for r in lr["image_regions"][:3] if isinstance(r,dict) and r.get("name")]))
        return "\n".join(ctx)

    def _kb_snippets(self, q:str): 
        return self.kb.query(q, k=2) if self.kb else ""

    def _append_cta(self, text: str) -> str:
        # cta = " Want me to show numbers, highlight the image, or try another what‑if?"
        # return text if text.strip().endswith("?") or cta in text else (text.strip() + "\n" + cta)
        return text 
    
    def _narrate_current_state(self, user_msg: str, state: Dict[str, Any], edits: dict | None = None) -> str:
        """Ask the LLM to explain what changed (or fallback to a friendly template)."""
        ctx = self._render_context(state)
        kb  = self._kb_snippets(user_msg)
        edit_line = f"\nAPPLIED_EDIT: {edits}" if edits else ""
        prompt_context = f"CONTEXT:\n{ctx}\n{kb}{edit_line}"

        if not self._llm_available():
            # Fallback: short, natural text + CTA
            base = state.get("last_result", {})
            gist = base.get("non_tech_summary") or base.get("text_narrative") or "Explanation updated."
            return self._append_cta(gist)

        client = openai.OpenAI() if hasattr(openai, "OpenAI") else openai
        sys = self.system_prompt
        try:
            if hasattr(client, "chat") and hasattr(client.chat, "completions"):
                resp = client.chat.completions.create(
                    model=self.model, temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": sys},
                        {"role": "user", "content": prompt_context},
                        {"role": "user", "content": f"User asked: {user_msg}\nPlease answer in natural language (Gist → Bridge → Next Action)."}
                    ],
                )
                text = resp.choices[0].message.content
            else:
                resp = client.responses.create(
                    model=self.model, temperature=self.temperature,
                    input=[
                        {"role": "system", "content": sys},
                        {"role": "user", "content": prompt_context},
                        {"role": "user", "content": f"User asked: {user_msg}\nPlease answer in natural language (Gist → Bridge → Next Action)."}
                    ],
                )
                text = resp.output_text
        except Exception as e:
            text = f"(LLM error fallback) {e}"
        return self._append_cta(text)


    def reply(self, user_msg:str, state:Dict[str,Any])->Tuple[str,Dict[str,Any]]:
        # what-if?
        # what-if?
        edits = parse_what_if(user_msg)
        if edits:
            base, img = state.get("last_user_input"), state.get("last_image_path")
            if not base or not img:
                return (self._append_cta("Please generate an explanation first."), state)

            # Convert relative directives to concrete numbers
            val = list(edits.values())[0]
            key = list(edits.keys())[0]
            if isinstance(val, tuple) and len(val) == 2 and val[1] == "relative":
                base_val = base.get(key)
                if key == "Reg_year" and isinstance(base_val, (int, float)):
                    delta = -1 if val[0] in {"decrease", "older"} else 1
                    edits = {key: int(base_val + delta)}
                elif isinstance(base_val, (int, float)):
                    step = float(getattr(self, "rel_step", 0.2))
                    factor = (1.0 - step) if val[0] in {"decrease","smaller","lower","older"} else (1.0 + step)
                    edits = {key: round(base_val * factor, 2)}
                else:
                    if key == "Gearbox":
                        return (self._append_cta("Gearbox is categorical (0=Automatic, 1=Manual). Try: 'what if gearbox is manual'."), state)
                    return (self._append_cta("I can only apply relative changes to numeric fields."), state)

            # Apply the what‑if to update panels/state
            updated = dict(base); updated.update(edits)
            cf = self.toolkit.run_counterfactual(updated, img)
            state["last_result"] = cf

            # Now ask the LLM to narrate (natural language) based on the updated state
            text = self._narrate_current_state(user_msg, state, edits=edits)
            return (text, state)


        if not state.get("last_result"):
            return (self._append_cta("Generate an explanation first, then ask me anything."), state)

        # LLM path or fallback
        if not self._llm_available():
            text = state["last_result"].get("non_tech_summary") or "Explanation is ready."
            kb = self._kb_snippets(user_msg)
            if kb: text += "\n(Knowledge) " + kb[:200]
            return (self._append_cta(text), state)

        client = openai.OpenAI() if hasattr(openai,"OpenAI") else openai
        ctx, kb = self._render_context(state), self._kb_snippets(user_msg)
        try:
            if hasattr(client,"chat") and hasattr(client.chat,"completions"):
                resp = client.chat.completions.create(
                    model=self.model, temperature=self.temperature,
                    messages=[{"role":"system","content":self.system_prompt},
                              {"role":"user","content":f"CONTEXT:\n{ctx}\n{kb}"},
                              {"role":"user","content":user_msg}]
                )
                text = resp.choices[0].message.content
            else:
                resp = client.responses.create(
                    model=self.model, temperature=self.temperature,
                    input=[{"role":"system","content":self.system_prompt},
                           {"role":"user","content":f"CONTEXT:\n{ctx}\n{kb}"},
                           {"role":"user","content":user_msg}]
                )
                text = resp.output_text
        except Exception as e:
            text = f"(LLM error fallback) {e}"
        return (self._append_cta(text), state)
