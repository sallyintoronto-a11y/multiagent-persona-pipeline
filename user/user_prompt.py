# ============================================================
# Prompts — minimize system, put all instructions into user
# ============================================================
MAX_VIEWPOINT_CHARS = 240
MAX_BIAS_CHARS = 240

def _clip(txt: str, n: int) -> str:
    return txt if len(txt) <= n else (txt[: n - 1] + "…")

SYSTEM_TMPL_MIN = (
    "You are a cultural persona vision agent.\n"
    "Speak in natural, everyday Korean as if you truly ARE the assigned persona (do not worry about being formal or academic)."
)

PERSONA_USER_TMPL_ALL = PromptTemplate.from_template(
    """CONTEXT
- Persona: {persona_description}
- Viewpoint: {persona_viewpoint}
- Bias_profile: {persona_bias_profile}

LANGUAGE: Korean

TASK
- Describe the image as if you were speaking casually in daily life.
- Do not worry about being formal or academic.
- Interpret the scene using your persona's viewpoint and bias_profile.
- After describing and interpreting, in "reasoning" add 1–2 sentences about how your own background (persona) may have influenced your interpretation.
- Keep visual claims image-grounded; mark conjecture as such.

FORMAT (return JSON only)
{{
  "agent_id": "{agent_id}",
  "image_refs": ["{image_filename}"],
  "visual_description": "...",
  "interpretation": "...",
  "cultural_or_ethical_reflection": "...",
  "reasoning": "..."
}}

[IMAGE]: {image_hint}
"""
)

def build_persona_system_prompt(_: Dict[str, Any]) -> str:
    return SYSTEM_TMPL_MIN

def build_persona_user_prompt(persona: Dict[str, Any], image_path: str) -> str:
    vp = _clip(persona.get("viewpoint", ""), MAX_VIEWPOINT_CHARS)
    bp_list = persona.get("bias_profile", [])
    bp = "; ".join(bp_list) if isinstance(bp_list, list) else str(bp_list)
    return PERSONA_USER_TMPL_ALL.format(
        persona_description=persona.get("description", ""),
        persona_viewpoint=vp,
        persona_bias_profile=bp,
        agent_id=persona.get("id", "agent"),
        image_filename=os.path.basename(image_path),
        image_hint=os.path.basename(image_path),
    )
