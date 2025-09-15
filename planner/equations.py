from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict
import json, os

"""
Xu et al. (Nature Communications, 2025): explicit symbolic relations connecting TLC Rf and solvent
ratio r = PE/(PE+EA) to mean CC start/end volumes V̄S, V̄E on silica for PE/EA systems.
This module exposes a pluggable form:

    V̄S = r / (a_s * Rf + b_s)     for r > 0
    V̄E = r / (a_e * Rf + b_e)     for r > 0

Edge case r==0 (pure EA) is handled via constants c_s0, c_e0 if provided.
Values are then scaled by a column_mass_scale factor relative to the reference (e.g., 4 g).

NOTE: The default coefficients in coeffs_default.json are **approximations** transcribed from the paper's
figures for the PE/EA system. You should calibrate them for your lab/column with a few runs.
"""

@dataclass
class XuCoeffs:
    # numerator: r
    a_s: float
    b_s: float
    a_e: float
    b_e: float
    c_s0: float = 5.15      # fallback for r=0 (pure EA) if applicable
    c_e0: float = 10.98     # fallback for r=0
    ref_silica_g: float = 4.0  # reference column silica mass corresponding to the coefficients

    @classmethod
    def from_json(cls, path: str) -> "XuCoeffs":
        with open(path, "r") as f:
            d = json.load(f)
        return cls(**d)

def load_default_coeffs() -> XuCoeffs:
    here = os.path.dirname(os.path.abspath(__file__))
    coeff_path = os.path.join(os.path.dirname(here), "data", "coeffs_default.json")
    return XuCoeffs.from_json(coeff_path)

def predict_vs_ve(r: float, rf: float, silica_g: float, coeffs: XuCoeffs | None = None) -> Tuple[float, float]:
    """
    Predict mean start/end volumes (mL) for given r (=PE/(PE+EA)) and TLC Rf.
    Scaled by silica mass relative to reference column.
    """
    coeffs = coeffs or load_default_coeffs()
    r = max(0.0, min(1.0, r))
    rf = max(0.0, min(0.99, rf))
    scale = max(0.05, silica_g / max(0.01, coeffs.ref_silica_g))  # linear scale with silica mass

    if r == 0.0:
        VS = coeffs.c_s0
        VE = coeffs.c_e0
    else:
        VS = r / (coeffs.a_s * rf + coeffs.b_s)
        VE = r / (coeffs.a_e * rf + coeffs.b_e)

    return VS * scale, VE * scale
