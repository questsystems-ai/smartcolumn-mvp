# planner/planner.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
from math import pi, ceil

from .equations import predict_vs_ve
from .utils import clamp


@dataclass
class PlannerConfig:
    # Load rule-of-thumb (STANDARD 20×; Cons 30×; Eff 15×)
    silica_ratio_x: float = 20.0                 # STANDARD load (× sample mass)
    silica_bulk_density_g_per_mL: float = 0.60   # packed silica bulk density
    porosity: float = 0.40                       # CV = ε × bed volume

    # Gradient pacing (STANDARD baseline; others scale by factors)
    step_size_pctEA: float = 5.0                 # %EA increment
    ramp_CV: float = 0.2                         # small transition per step (in CV)
    inc_CV_standard: float = 2.0                 # STANDARD increment/pre-eq = 2 CV
    inc_CV_conservative_factor: float = 1.5      # 🛡️ → 3.0 CV
    inc_CV_efficient_factor: float = 0.75        # ⚡  → 1.5 CV
    final_plateau_factor: float = 3.0            # final plateau = 3 × increment

    # Composition bounds
    max_pctEA: float = 70.0

    # Flow default
    default_cv_per_min: float = 0.5              # 0.5 CV/min (~2 min per CV)

    # Geometry (12 cm target bed; allow ±2 cm)
    bed_height_target_cm: float = 12.0
    bed_height_tol_cm: float = 2.0

    # Final %EA heuristic (stay near TLC & keep VE in 2–6 CV)
    target_VE_CV_lo: float = 2.0
    target_VE_CV_hi: float = 6.0
    final_pctEA_window_from_tlc: float = 10.0    # clamp final %EA to TLC±10%


@dataclass
class ColumnPlan:
    # Identity
    mode: str
    vendor_name: Optional[str]
    cartridge_name: Optional[str]

    # Bulk & geometry
    silica_g: float
    glass_id_cm: float
    bed_height_cm: float
    packed_volume_mL: float
    column_volume_mL: float

    # Ops
    flow_mL_min: float
    tlc_pctEA: float
    predicted_elution_mL: Tuple[float, float]       # at final isocratic
    predicted_elution_tlc_mL: Tuple[float, float]   # at TLC system

    # Program
    pre_equilibrate_mL: float
    increment_size_pctEA: float
    increment_volume_mL: float
    final_pctEA: float
    final_plateau_mL: float
    total_solvent_mL: float
    total_time_min: float


# ----------------- helpers -----------------

def _required_bed_height_cm(silica_g: float, id_cm: float, rho_g_per_mL: float) -> float:
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
    allowing ±2 cm. Prefer solutions within [target−tol, target+tol].
    Returns (ID, bed_h_used, packed_volume_mL).
    """
    best_id = None
    best_h = None
    best_err = 1e9
    in_tol: List[Tuple[float, float, float]] = []

    for ID in standard_ids_cm:
        h_req = _required_bed_height_cm(silica_g, ID, rho)
        err = abs(h_req - target_h)
        if (target_h - tol_h) <= h_req <= (target_h + tol_h):
            in_tol.append((ID, h_req, err))
        if err < best_err:
            best_err, best_id, best_h = err, ID, h_req

    if in_tol:
        in_tol.sort(key=lambda x: x[2])  # closest to target within tolerance
        ID, h_used, _ = in_tol[0]
    else:
        ID, h_used = best_id, best_h

    packed_vol_mL = pi * (ID / 2.0) ** 2 * h_used
    return ID, h_used, packed_vol_mL


def _choose_final_pctEA_multi(
    rf: float, silica_g: float, cv_mL: float, cfg: PlannerConfig, tlc_pctEA: float
) -> float:
    """
    Multi-objective final %EA selector within [TLC±window]:
      minimize  w1*|pct - TLC| + w2*|VE_CV - 4| + w3*steps
    where steps = final%/step_size, VE_CV = VE / CV, and 4 CV is the middle of the 2–6 band.
    """
    w1, w2, w3 = 1.0, 2.0, 0.1
    lo = max(0, int(round(tlc_pctEA - cfg.final_pctEA_window_from_tlc)))
    hi = min(int(cfg.max_pctEA), int(round(tlc_pctEA + cfg.final_pctEA_window_from_tlc)))

    best_pct = None
    best_score = 1e9
    for pct in range(lo, hi + 1, int(cfg.step_size_pctEA)):
        r = 1.0 - pct / 100.0
        _, VE = predict_vs_ve(r, rf, silica_g)
        VE_CV = VE / max(1e-6, cv_mL)
        steps = max(1, int(round(pct / cfg.step_size_pctEA)))
        score = w1 * abs(pct - tlc_pctEA) + w2 * abs(VE_CV - 4.0) + w3 * steps
        if score < best_score:
            best_score, best_pct = score, pct

    return float(best_pct if best_pct is not None else int(round(tlc_pctEA)))


# ----------------- main API -----------------

def plan_column(
    rf: float,
    tlc_pctEA: float,             # TLC system (%EA)
    mass_mg: float,
    standard_ids_cm: List[float], # used for hand-column geometry search
    cfg: PlannerConfig | None = None,
    mode: str = "standard",
    override_id_cm: float | None = None,          # hand-column: force specific ID
    silica_override_g: float | None = None,       # autocolumn: cartridge silica
    preeq_CV_override: float | None = None,       # vendor equilibration CV (e.g., 2.0)
    vendor_name: Optional[str] = None,
    cartridge_name: Optional[str] = None,
) -> ColumnPlan:
    """
    1) silica from mass (STANDARD 20×), scaled by mode, unless silica_override_g is provided
    2) choose ID to keep bed near 12 cm (±2 cm); or honor override_id_cm
    3) CV = ε × bed volume; final %EA near TLC by multi-objective
    4) program: pre-eq & increment in CV; final plateau = 3× increment (pre-eq CV may be overridden)
    """
    assert mode in ("conservative", "standard", "efficient")
    cfg = cfg or PlannerConfig()
    rf = clamp(rf, 0.0, 0.99)
    tlc_pctEA = clamp(tlc_pctEA, 0.0, cfg.max_pctEA)

    # 1) silica load (mode scaling) — or override
    if silica_override_g is not None:
        silica_g = max(0.01, float(silica_override_g))
    else:
        silica_scale = 1.5 if mode == "conservative" else (0.75 if mode == "efficient" else 1.0)
        silica_g = (mass_mg / 1000.0) * cfg.silica_ratio_x * silica_scale

    # 2) geometry: pick ID/height (or compute from override)
    if override_id_cm is not None:
        bed_h_cm = _required_bed_height_cm(silica_g, override_id_cm, cfg.silica_bulk_density_g_per_mL)
        glass_id_cm = float(override_id_cm)
        packed_vol_mL = pi * (glass_id_cm / 2.0) ** 2 * bed_h_cm
    else:
        glass_id_cm, bed_h_cm, packed_vol_mL = _choose_id_and_bed_height(
            silica_g=silica_g,
            rho=cfg.silica_bulk_density_g_per_mL,
            target_h=cfg.bed_height_target_cm,
            tol_h=cfg.bed_height_tol_cm,
            standard_ids_cm=standard_ids_cm,
        )

    # 3) CV and flow
    cv_mL = cfg.porosity * packed_vol_mL
    flow_mL_min = max(0.1, cv_mL * cfg.default_cv_per_min)

    # 4) final %EA near TLC
    final_pctEA = _choose_final_pctEA_multi(rf, silica_g, cv_mL, cfg, tlc_pctEA)
    increment_size_pctEA = cfg.step_size_pctEA

    # increments per mode (CV)
    if mode == "conservative":
        inc_CV = cfg.inc_CV_standard * cfg.inc_CV_conservative_factor  # 3.0
    elif mode == "efficient":
        inc_CV = cfg.inc_CV_standard * cfg.inc_CV_efficient_factor     # 1.5
    else:
        inc_CV = cfg.inc_CV_standard                                   # 2.0

    increment_volume_mL = inc_CV * cv_mL
    pre_eq_mL = (preeq_CV_override * cv_mL) if (preeq_CV_override is not None) else increment_volume_mL
    final_plateau_mL = cfg.final_plateau_factor * increment_volume_mL

    # total ladder: 0 → final_%EA in step_size increments
    steps = max(1, int(ceil(final_pctEA / increment_size_pctEA)))
    ramp_mL = cfg.ramp_CV * cv_mL
    ladder_mL = increment_volume_mL + max(0, steps - 1) * (ramp_mL + increment_volume_mL)
    total_mL = pre_eq_mL + ladder_mL + final_plateau_mL
    total_time_min = total_mL / flow_mL_min

    # predictions (display)
    r_fin = 1.0 - final_pctEA / 100.0
    vs_fin, ve_fin = predict_vs_ve(r_fin, rf, silica_g)
    r_tlc = 1.0 - tlc_pctEA / 100.0
    vs_tlc, ve_tlc = predict_vs_ve(r_tlc, rf, silica_g)

    return ColumnPlan(
        mode=mode,
        vendor_name=vendor_name,
        cartridge_name=cartridge_name,
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
