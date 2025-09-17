from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import json, os

@dataclass
class Cartridge:
    name: str
    silica_g: float
    particle_um: str | None = None

@dataclass
class Vendor:
    name: str
    equil_CV_default: float
    export: Dict[str, bool]
    cartridges: List[Cartridge]
    load_bands: Dict[str, float]  # easy_pct / typical_pct / hard_pct

def load_vendors(json_path: Optional[str] = None) -> List[Vendor]:
    here = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(os.path.dirname(here), "data", "vendors.json")
    path = json_path or default_path
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: List[Vendor] = []
    for v in raw["vendors"]:
        carts = [Cartridge(**c) for c in v.get("cartridges", [])]
        out.append(Vendor(
            name=v["name"],
            equil_CV_default=float(v.get("equil_CV_default", 2.0)),
            export=v.get("export", {}),
            cartridges=carts,
            load_bands=v.get("load_bands", {})
        ))
    return out
