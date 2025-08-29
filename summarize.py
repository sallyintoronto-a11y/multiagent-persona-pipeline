# === Summaries to JSON only + Download to local ===
from __future__ import annotations
import os, glob, json, zipfile
from pathlib import Path
from typing import Dict, Any, Tuple, List

# ---- CONFIG ----
OUT_DIR = "/content/outputs"     # "./outputs_userprompt"
ROUND_DIGITS = 3

def _safe_load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _agent_axes(a: Dict[str, Any]) -> Tuple[float, float, float]:
    return (
        float(a.get("factuality", 0)),
        float(a.get("norm_appropriateness", 0)),
        float(a.get("bias_presence", 0)),
    )

def summarize_pair(pair_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    pair_result example:
    {
      "pair": "GENDER",
      "evaluation": { "agents": {...}, "delta": {...} },
      "agent_outputs": {...}
    }
    """
    eval_block = pair_result.get("evaluation", {})
    agents = eval_block.get("agents", {}) or {}

    agent_overall: Dict[str, float] = {}
    fx, nx, bx = [], [], []

    for aid, score in agents.items():
        f, n, b = _agent_axes(score)
        fx.append(f); nx.append(n); bx.append(b)
        # overall = f + n + b (bias_presence is a "no bias" score, so should be added)
        agent_overall[aid] = f + n + b

    if not fx:
        axis_avg = {"factuality": 0.0, "norm_appropriateness": 0.0, "bias_presence": 0.0}
    else:
        axis_avg = {
            "factuality": round(sum(fx)/len(fx), ROUND_DIGITS),
            "norm_appropriateness": round(sum(nx)/len(nx), ROUND_DIGITS),
            "bias_presence": round(sum(bx)/len(bx), ROUND_DIGITS),
        }

    # pair_overall also computed as f + n + b
    pair_overall = round(
        axis_avg["factuality"] + axis_avg["norm_appropriateness"] + axis_avg["bias_presence"],
        ROUND_DIGITS
    )

    return {
        "axis_avg": axis_avg,
        "pair_overall": pair_overall,
        "agents_overall": {k: round(v, ROUND_DIGITS) for k, v in agent_overall.items()},
    }

# ---- Build summaries from outputs/by_image/*.json ----
records: List[Dict[str, Any]] = []           # Flat format (JSON array)
master: Dict[str, Dict[str, Any]] = {}       # master[image][pair] = summary

by_image_dir = os.path.join(OUT_DIR, "by_image")
paths = sorted(glob.glob(os.path.join(by_image_dir, "*.json")))
if not paths:
    raise FileNotFoundError(f"No per-image reports found in: {by_image_dir}")

for path in paths:
    data = _safe_load_json(path)
    # If image_path exists, use that filename; if not, use filename stem as image name
    image_name = os.path.basename((data or {}).get("image_path", "")) or os.path.splitext(os.path.basename(path))[0]
    master.setdefault(image_name, {})

    for pr in (data or {}).get("pairs", []):
        pair_key = pr.get("pair", "UNKNOWN")
        summary = summarize_pair(pr)
        master[image_name][pair_key] = summary

        # pair-avg one row
        records.append({
            "image": image_name,
            "pair": pair_key,
            "type": "pair_avg",
            "agent_id": "",
            "factuality": summary["axis_avg"]["factuality"],
            "norm_appropriateness": summary["axis_avg"]["norm_appropriateness"],
            "bias_presence": summary["axis_avg"]["bias_presence"],
            "overall": summary["pair_overall"],
        })
        # one row per agent
        agents_scores = (pr.get("evaluation", {}).get("agents", {}) or {})
        for aid, ov in summary["agents_overall"].items():
            f, n, b = _agent_axes(agents_scores.get(aid, {}))
            records.append({
                "image": image_name,
                "pair": pair_key,
                "type": "agent",
                "agent_id": aid,
                "factuality": f,
                "norm_appropriateness": n,
                "bias_presence": b,
                "overall": ov,
            })

# ---- Save JSONs ----
summary_dir = os.path.join(OUT_DIR, "summary")   # "summary_user"
Path(summary_dir).mkdir(parents=True, exist_ok=True)

master_json_path = os.path.join(summary_dir, "summary_master.json")   # "summary_master_user.json"
with open(master_json_path, "w", encoding="utf-8") as f:
    json.dump(master, f, ensure_ascii=False, indent=2)

flat_json_path = os.path.join(summary_dir, "summary_pairs.json")   # "summary_pairs_user.json"
with open(flat_json_path, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"Saved master JSON  → {master_json_path}")
print(f"Saved flat JSON    → {flat_json_path}")
