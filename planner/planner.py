# planner/planner.py
from __future__ import annotations
from typing import Dict, List, Tuple
from dataclasses import dataclass
from math import pi

from .equations import predict_vs_ve
from .utils import clamp


@dataclass
class PlannerConfig:
    # Load rule-of-thumb
    silica_ratio_x: float = 40.0             # silica : sample mass (x)
    silica_bulk_density_g_per_mL: float = 0.60  # packed silica bulk density (≈0.55–0.65)
    porosity: float = 0.40                   # interstitial fraction ε → CV = ε * bed volume

    # Gradient pacing (STANDARD baseline; others scale by factors)
    step_size_pctEA: float = 5.0             # %EA increment (final% over 5)
    ramp_CV: float = 0.2                     # small transition per step (in CV)
    inc_CV_standard: float = 2.0             # STANDARD increments/pre-eq = 2 CV
    inc_CV_conservative_factor: float = 1.5  # 🛡️  3.0 CV
    inc_CV_efficient_factor: float = 0.75    # ⚡  1.5 CV
    final_plateau_factor: float = 3.0        # final plateau = 3 × increment

    # Composition bounds
    max_pctEA: float = 70.0                  # EA cap for PE/EA scope

    # Flow default
    default_cv_per_min: float = 0.5          # 0.5 CV/min (~2 min per CV)

    # Geometry (12 cm target bed; allow ±2 cm if that helps)
    bed_height_target_cm: float = 12.0
    bed_height_tol_cm: float = 2.0

    # Final %EA heuristic: pick earliest with V̄E in 2–6 CV
    target_VE_CV_lo: float = 2.0
    target_VE_CV_hi: float = 6.0


@dataclass
class ColumnPlan:
    mode: str
    silica_g: float
    glass_id_cm: float
    bed_height_cm: float
    packed_volume_mL: float             # bed volume (mL == cm^3)
    column_volume_mL: float             # CV = ε * bed volume
    flow_mL_min: float
    tlc_pctEA: float
    predicted_elution_mL: Tuple[float, float]      # at final isocratic
    predicted_elution_tlc_mL: Tuple[float, float]  # at TLC system (context)
    # Summary-only program:
    pre_equilibrate_mL: float
    increment_size_pctEA: float
    increment_volume_mL: float
    final_pctEA: float
    final_plateau_mL: float
    total_solvent_mL: float
    total_time_min: float


# -------- helpers --------

def _required_bed_height_cm(silica_g: float, id_cm: float, rho_g_per_mL: float) -> float:
    """Bed height required to hold 'silica_g' at bulk density in a column of ID 'id_cm'."""
    packed_vol_cm3 = silica_g / max(1e-6, rho_g_per_mL)
    area_cm2 = pi * (id_cm / 2.0) ** 2
    return packed_vol_cm3 / max(1e-6, area_cm2)


def _choose_id_and_bed_height(
    silica_g: float,
    rho: float,
    target_h: float,
    tol_h: float,
    standard_ids_cm: List[float]
) -> Tuple[float, float, float]:
    """
    Choose an ID so bed height is as close as possible to target (12 cm),
    allowing ±2 cm adjustment. Prefer solutions within [target−tol, target+tol].
    Returns (ID, bed_h_used, packed_volume_mL).
    """
    best_id = None
    best_h = None
    best_err = 1e9
    in_tol_candidates: List[Tuple[float, float, float]] = []

    for ID in standard_ids_cm:
        h_req = _required_bed_height_cm(silica_g, ID, rho)
        err = abs(h_req - target_h)
        if (target_h - tol_h) <= h_req <= (target_h + tol_h):
            in_tol_candidates.append((ID, h_req, err))
        if err < best_err:
            best_err, best_id, best_h = err, ID, h_req

    if in_tol_candidates:
        in_tol_candidates.sort(key=lambda x: x[2])  # closest to target within tol
        ID, h_used, _ = in_tol_candidates[0]
    else:
        # No ID yields bed within ±2 cm; pick nearest achievable
        ID, h_used = best_id, best_h

    packed_vol_mL = pi * (ID / 2.0) ** 2 * h_used  # mL == cm^3
    return ID, h_used, packed_vol_mL


def _choose_final_pctEA(rf: float, silica_g: float, cv_mL: float, cfg: PlannerConfig) -> float:
    for pct in range(0, int(cfg.max_pctEA) + 1, int(cfg.step_size_pctEA)):
        r = 1.0 - pct / 100.0
        _, VE = predict_vs_ve(r, rf, silica_g)
        VE_CV = VE / max(1e-6, cv_mL)
        if cfg.target_VE_CV_lo <= VE_CV <= cfg.target_VE_CV_hi:
            return float(pct)
    return 50.0  # fallback


# -------- main API --------

def plan_column(
    rf: float,
    tlc_pctEA: float,        # TLC system (%EA) – required to contextualize Rf
    mass_mg: float,
    standard_ids_cm: List[float],  # pass standard column IDs from the UI
    cfg: PlannerConfig | None = None,
    mode: str = "standard",
) -> ColumnPlan:
    """
    1) silica from mass (x-ratio), with mode scaling: 🛡️ 1.5×, ⚖️ 1.0×, ⚡ 0.75×
    2) pick standard ID to keep bed near 12 cm (±2 cm allowed); otherwise nearest achievable
    3) CV = ε * bed volume; final %EA via oracle (V̄E in 2–6 CV)
    4) summary program: pre-eq & increment in CV (STANDARD=2 CV; 🛡️=3; ⚡=1.5), final plateau = 3× increment
    """
    assert mode in ("conservative", "standard", "efficient")
    cfg = cfg or PlannerConfig()
    rf = clamp(rf, 0.0, 0.99)
    tlc_pctEA = clamp(tlc_pctEA, 0.0, cfg.max_pctEA)

    # 1) silica (rule-of-thumb) with mode scaling
    silica_scale = 1.5 if mode == "conservative" else (0.75 if mode == "efficient" else 1.0)
    silica_g = (mass_mg / 1000.0) * cfg.silica_ratio_x * silica_scale

    # 2) choose ID & bed height (12 cm ± 2 cm if possible)
    glass_id_cm, bed_h_cm, packed_vol_mL = _choose_id_and_bed_height(
        silica_g=silica_g,
        rho=cfg.silica_bulk_density_g_per_mL,
        target_h=cfg.bed_height_target_cm,
        tol_h=cfg.bed_height_tol_cm,
        standard_ids_cm=standard_ids_cm
    )

    # 3) CV and flow
    cv_mL = cfg.porosity * packed_vol_mL
    flow_mL_min = max(0.1, cv_mL * cfg.default_cv_per_min)

    # 4) final %EA via oracle
    final_pctEA = _choose_final_pctEA(rf, silica_g, cv_mL, cfg)
    increment_size_pctEA = final_pctEA / 5.0

    # increments per mode (CV)
    if mode == "conservative":
        inc_CV = cfg.inc_CV_standard * cfg.inc_CV_conservative_factor  # 3.0
    elif mode == "efficient":
        inc_CV = cfg.inc_CV_standard * cfg.inc_CV_efficient_factor     # 1.5
    else:
        inc_CV = cfg.inc_CV_standard                                    # 2.0

    increment_volume_mL = inc_CV * cv_mL
    pre_eq_mL = increment_volume_mL
    final_plateau_mL = cfg.final_plateau_factor * increment_volume_mL

    # steps 0→final in 5% increments
    steps = max(1, int(round(final_pctEA / cfg.step_size_pctEA + 1e-6)))
    ramp_mL = cfg.ramp_CV * cv_mL
    ladder_mL = increment_volume_mL + max(0, steps - 1) * (ramp_mL + increment_volume_mL)
    total_mL = pre_eq_mL + ladder_mL + final_plateau_mL
    total_time_min = total_mL / flow_mL_min

    # predictions
    r_fin = 1.0 - final_pctEA / 100.0
    vs_fin, ve_fin = predict_vs_ve(r_fin, rf, silica_g)

    r_tlc = 1.0 - tlc_pctEA / 100.0
    vs_tlc, ve_tlc = predict_vs_ve(r_tlc, rf, silica_g)

    return ColumnPlan(
        mode=mode,
        silica_g=silica_g,
        glass_id_cm=glass_id_cm,
        bed_height_cm=bed_h_cm,
        packed_volume_mL=packed_vol_mL,
        column_volume_mL=cv_mL,
        flow_mL_min=flow_mL_min,
        tlc_pctEA=tlc_pctEA,
        predicted_elution_mL=(vs_fin, ve_fin),
        predicted_elution_tlc_mL=(vs_tlc, ve_tlc),
        pre_equilibrate_mL=pre_eq_mL,
        increment_size_pctEA=increment_size_pctEA,
        increment_volume_mL=increment_volume_mL,
        final_pctEA=final_pctEA,
        final_plateau_mL=final_plateau_mL,
        total_solvent_mL=total_mL,
        total_time_min=total_time_min,
    )
