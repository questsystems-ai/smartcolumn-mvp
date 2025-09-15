from __future__ import annotations
from typing import Tuple, List, Dict

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def interleave_plateaus_and_ramps(steps: List[dict], ramp_cv: float, column_volume_mL: float) -> List[dict]:
    """Insert short linear ramp segments between plateaus for readability."""
    out = []
    for i, st in enumerate(steps):
        out.append({"type": "plateau", **st})
        if i < len(steps) - 1:
            out.append({
                "type": "ramp",
                "from_pctEA": steps[i]["pctEA"],
                "to_pctEA": steps[i+1]["pctEA"],
                "volume_mL": ramp_cv * column_volume_mL
            })
    return out

def warn_range_rf(rf: float) -> str | None:
    if rf < 0.1:
        return "Rf < 0.10 on TLC suggests strong retention — consider increasing %EA or using smaller load."
    if rf > 0.7:
        return "Rf > 0.70 on TLC suggests weak retention — consider decreasing %EA or starting at lower %EA."
    return None
