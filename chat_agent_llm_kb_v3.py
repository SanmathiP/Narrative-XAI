# chat_agent_llm_kb_v3.py
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
from narrative_engine_protoC import build_prompt  # NEW

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
        try: return float(raw) if "." in raw else int(float(raw))
        except Exception: return None
    if key == "Gearbox":
        if raw in {"auto","automatic","0"}: return 0
        if raw in {"manual","1"}: return 1
        return None
    return raw

def _find_feature_in_text(msg: str) -> Optional[str]:
    for alias, key in _FEATURE_ALIASES.items():
        if alias in msg: return key
    return None

def parse_what_if(message: str) -> Optional[Dict[str, Any]]:
    msg = message.lower()
    m = re.search(r"what\s*-?\s*if\s+([a-z_ ]+)\s*(?:is|=|to)\s*([-+\w\.]+)", msg)
    if m:
        feat_raw, val_raw = m.group(1).strip(), m.group(2).strip()
        feat_key = _FEATURE_ALIASES.get(feat_raw.replace(" ","_")) or _FEATURE_ALIASES.get(feat_raw) or feat_raw
        val = _cast_value(feat_key, val_raw)
        return {feat_key: val} if val is not None else None
    if any(w in msg for w in ["increase","increased","decrease","decreased","higher","lower","bigger","larger","smaller","older","newer"]):
        key = _find_feature_in_text(msg)
        if not key: return None
        return {key: ("decrease","relative")} if any(w in msg for w in ["decrease","decreased","smaller","lower","older"]) else {key: ("increase","relative")}
    return None

# ---------- toolkit ----------
@dataclass
class Toolkit:
    run_baseline: Callable[[Dict[str, Any], str], Dict[str, Any]]
    run_counterfactual: Callable[[Dict[str, Any], str], Dict[str, Any]]

class ChatAgent:
    def __init__(self, toolkit: Toolkit, model="gpt-4o-mini", temperature=0.2, kb_paths: List[str]|None=None, rel_step: float = 0.2, story_mode: bool = False):
        self.toolkit = toolkit
        self.model = model
        self.temperature = temperature
        self.kb = KnowledgeBase(paths=kb_paths or [])
        self.rel_step = rel_step
        self.story_mode = story_mode  # NEW

    # allow UI to toggle at runtime
    def set_story_mode(self, flag: bool):
        self.story_mode = bool(flag)

    def _llm_available(self): 
        return _HAS_OPENAI and bool(os.environ.get("OPENAI_API_KEY"))

    def _render_context(self, state: Dict[str,Any]) -> str:
        lr = state.get("last_result") or {}
        ctx = []
        if lr.get("non_tech_summary"): ctx.append("summary:"+lr["non_tech_summary"][:200])
        if lr.get("text_narrative"):   ctx.append("narrative:"+lr["text_narrative"][:200])
        if lr.get("top_features"):     ctx.append("top_features:"+",".join([f["name"] for f in lr["top_features"][:5] if isinstance(f,dict) and f.get("name")]))
        if lr.get("image_regions"):    ctx.append("image_focus:"+",".join([r["name"] for r in lr["image_regions"][:3] if isinstance(r,dict) and r.get("name")]))
        return "\n".join(ctx)

    def _kb_snippets(self, q:str): 
        return self.kb.query(q, k=2) if self.kb else ""

    # central LLM call using narrative_engine
    def _ask_llm(self, user_msg: str, context: str, knowledge: str) -> str:
        prompt = build_prompt(context=context, knowledge=knowledge, question=user_msg, mode="business_story" if self.story_mode else "concise")
        if not self._llm_available():
            # graceful fallback: return the prompt’s first line as hint
            return "Explanation updated. (LLM unavailable)"
        client = openai.OpenAI() if hasattr(openai,"OpenAI") else openai
        try:
            if hasattr(client,"chat") and hasattr(client.chat,"completions"):
                resp = client.chat.completions.create(
                    model=self.model, temperature=self.temperature,
                    messages=[{"role":"system","content":"Follow the prompt exactly. Never invent numbers."},
                              {"role":"user","content":prompt}],
                )
                return resp.choices[0].message.content
            else:
                resp = client.responses.create(
                    model=self.model, temperature=self.temperature,
                    input=[{"role":"system","content":"Follow the prompt exactly. Never invent numbers."},
                           {"role":"user","content":prompt}],
                )
                return resp.output_text
        except Exception as e:
            return f"(LLM error: {e})"

    def reply(self, user_msg:str, state:Dict[str,Any])->Tuple[str,Dict[str,Any]]:
        # what-if?
        edits = parse_what_if(user_msg)
        if edits:
            base, img = state.get("last_user_input"), state.get("last_image_path")
            if not base or not img:
                return ("Please generate an explanation first.", state)
            val = list(edits.values())[0]; key = list(edits.keys())[0]
            if isinstance(val, tuple) and len(val)==2 and val[1]=="relative":
                base_val = base.get(key)
                if key == "Reg_year" and isinstance(base_val,(int,float)):
                    delta = -1 if val[0] in {"decrease","older"} else 1
                    edits = {key: int(base_val + delta)}
                elif isinstance(base_val,(int,float)):
                    step = float(getattr(self, "rel_step", 0.2))
                    factor = (1.0 - step) if val[0] in {"decrease","smaller","lower","older"} else (1.0 + step)
                    edits = {key: round(base_val * factor, 2)}
                else:
                    if key=="Gearbox":
                        return ("Gearbox is categorical (0=Automatic, 1=Manual). Try: 'what if gearbox is manual'.", state)
                    return ("I can only apply relative changes to numeric fields.", state)
            updated = dict(base); updated.update(edits)
            cf = self.toolkit.run_counterfactual(updated, img)
            state["last_result"] = cf
            # narrate naturally (story or concise)
            ctx = self._render_context(state)
            kb  = self._kb_snippets(user_msg)
            text = self._ask_llm(user_msg, ctx, kb)
            return (text, state)

        if not state.get("last_result"):
            return ("Generate an explanation first, then ask me anything.", state)

        ctx = self._render_context(state)
        kb  = self._kb_snippets(user_msg)
        text = self._ask_llm(user_msg, ctx, kb)
        return (text, state)
