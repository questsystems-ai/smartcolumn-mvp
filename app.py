# app.py
from __future__ import annotations

# ---- MUST be the first Streamlit call ----
import streamlit as st
st.set_page_config(page_title="SmartColumn", page_icon="🧪", layout="wide")

# ---- Imports (no UI at import-time) ----
from dataclasses import asdict
import pandas as pd
import streamlit.components.v1 as components
from pathlib import Path

from planner.planner import plan_column, PlannerConfig
from post_run import render_post_run
from db_supabase import get_sb  # loads .env locally and st.secrets in cloud

# ----------------- Supabase init (no UI here) -----------------
sb = None
_sb_err = None
try:
    sb = get_sb()
except Exception as e:
    _sb_err = str(e)

# ----------------- Router (session-state only) -----------------
if "route" not in st.session_state:
    st.session_state["route"] = "plan"

def goto(route: str, **stash):
    # Stash any context, set route, and rerun
    for k, v in stash.items():
        st.session_state[k] = v
    st.session_state["route"] = route
    st.rerun()

# ----------------- Shared constants & helpers -----------------
STANDARD_GLASS_IDS_CM = [x / 2 for x in range(1, 25)]  # 0.5 .. 12.0

COMBIFLASH_CARTS = [
    {"name": "RediSep Rf 4 g",   "silica_g": 4},
    {"name": "RediSep Rf 12 g",  "silica_g": 12},
    {"name": "RediSep Rf 24 g",  "silica_g": 24},
    {"name": "RediSep Rf 40 g",  "silica_g": 40},
    {"name": "RediSep Rf 80 g",  "silica_g": 80},
    {"name": "RediSep Rf 160 g", "silica_g": 160},
    {"name": "RediSep Rf 330 g", "silica_g": 330},
]

# Rough cost placeholders
PRICE_PER_24L_USD  = 120.0
DISPOSAL_PER_L_USD = 6.0
SILICA_COST_PER_KG = 100.0

# Solvent names for PE/EA system used in TLC
SOLVENT_A_NAME = "Hexane (PE)"
SOLVENT_B_NAME = "Ethyl acetate (EA)"

def mode_label(mode: str) -> str:
    return ("🛡️ Conservative" if mode == "conservative"
            else "⚡ Efficient" if mode == "efficient"
            else "⚖️ Standard")

def cart_for_costing(silica_g: float):
    for c in COMBIFLASH_CARTS:
        if silica_g <= c["silica_g"]:
            price = {4: 25, 12: 39, 24: 59, 40: 79, 80: 119, 160: 179, 330: 259}.get(c["silica_g"], 0)
            return c["name"], price
    last = COMBIFLASH_CARTS[-1]
    return last["name"], 259

def show_mathjax_html(path: str, height: int = 900):
    html = Path(path).read_text(encoding="utf-8")
    components.html(html, height=height, scrolling=True)

def highlight_block(plan, autocolumn: bool):
    """Top summary strip. In autocolumn mode, do NOT show 'increment'."""
    # Define shared pieces first so they're always in scope
    final_iso = f"{round(plan.final_pctEA)}% EA"

    if autocolumn:
        # Rounded display for readability
        preeq_disp = int(round(plan.pre_equilibrate_mL / 10.0) * 10)
        flow_disp = int(round(plan.flow_mL_min))

        top_line = (
            f"<b>Vendor:</b> {plan.vendor_name or 'CombiFlash Rf+'} &nbsp;|&nbsp; "
            f"<b>Cartridge:</b> {plan.cartridge_name or '—'}"
        )
        mid = (
            f"<b>Final isocratic:</b> {final_iso} &nbsp;|&nbsp; "
            f"<b>Equilibrate:</b> {preeq_disp} mL &nbsp;|&nbsp; "
            f"<b>Flow:</b> {flow_disp} mL/min"
        )
    else:
        size = f"{round(plan.glass_id_cm, 1)} cm ID × {round(plan.bed_height_cm)} cm bed"
        top_line = f"<b>Column:</b> {size}"
        inc_vol = f"{round(plan.increment_volume_mL)} mL"
        preeq = f"{round(plan.pre_equilibrate_mL)} mL"
        flow = f"{round(plan.flow_mL_min, 1)} mL/min"
        mid = (
            f"<b>Final isocratic:</b> {final_iso} &nbsp;|&nbsp; "
            f"<b>%EA increment:</b> 5% &nbsp;|&nbsp; "
            f"<b>Increment volume:</b> {inc_vol} &nbsp;|&nbsp; "
            f"<b>Pre-equilibration:</b> {preeq} &nbsp;|&nbsp; "
            f"<b>Flow:</b> {flow}"
        )

    st.markdown(
        f"""
        <div style="border:1px solid #ddd;padding:10px;border-radius:8px;background:#f9f9f9;">
        {top_line} &nbsp;|&nbsp; {mid}
        </div>
        """,
        unsafe_allow_html=True,
    )

def details_tables(plan):
    """Generic details + costs (used mainly in hand-column mode)."""
    method_rows = [
        ("Vendor",                         plan.vendor_name or "—"),
        ("Cartridge",                      plan.cartridge_name or "—"),
        ("TLC %EA",                        f"{round(plan.tlc_pctEA)}"),
        ("Silica (g)",                     f"{round(plan.silica_g)}"),
        ("Packed bed vol (mL)",            f"{round(plan.packed_volume_mL)}"),
        ("CV (mL)",                        f"{round(plan.column_volume_mL)}"),
        ("Increment size (%EA)",           f"{5}"),
        ("Final isocratic (%EA)",          f"{round(plan.final_pctEA)}"),
        ("Final plateau (mL)",             f"{round(plan.final_plateau_mL)}"),
        ("Predicted V̄S at TLC (mL)",      f"{round(plan.predicted_elution_tlc_mL[0])}"),
        ("Predicted V̄E at TLC (mL)",      f"{round(plan.predicted_elution_tlc_mL[1])}"),
        ("Predicted V̄S final (mL)",       f"{round(plan.predicted_elution_mL[0])}"),
        ("Predicted V̄E final (mL)",       f"{round(plan.predicted_elution_mL[1])}"),
        ("Total solvent (mL)",             f"{round(plan.total_solvent_mL)}"),
        ("Total time (min)",               f"{round(plan.total_time_min)}"),
    ]
    st.table(pd.DataFrame(method_rows, columns=["Parameter", "Value"]))

    cart_name, cart_price = cart_for_costing(plan.silica_g)
    solvent_cost  = round((plan.total_solvent_mL / 24000.0) * PRICE_PER_24L_USD)
    disposal_cost = round((plan.total_solvent_mL / 1000.0) * DISPOSAL_PER_L_USD)
    silica_cost   = round((plan.silica_g / 1000.0) * SILICA_COST_PER_KG)
    cost_rows = [
        ("Loose silica (g)",            f"{round(plan.silica_g)}"),
        ("Loose silica cost (USD)",     f"{silica_cost}"),
        ("Suggested cartridge",         cart_name),
        ("Cartridge cost (USD)",        f"{cart_price}"),
        ("Solvent (mL)",                f"{round(plan.total_solvent_mL)}"),
        ("Solvent purchase cost (USD)", f"{solvent_cost}"),
        ("Disposal cost (USD)",         f"{disposal_cost}"),
    ]
    st.table(pd.DataFrame(cost_rows, columns=["Cost Item", "Value"]))

def render_combiflash_method(plan):
    """Show a vendor-ready CombiFlash method: continuous gradient points (no step increments)."""
    cv_mL = float(plan.column_volume_mL)
    flow = float(plan.flow_mL_min)

    # Display rounding: flow → nearest 1; equilibration → nearest 10
    flow_disp = int(round(flow))
    preeq_disp = int(round(plan.pre_equilibrate_mL / 10.0) * 10)

    time_per_cv_min = cv_mL / max(1e-6, flow)

    # Continuous ramp: 0 %B at 0 CV → final %B at ~3.5 CV, then a short hold (from plan)
    ramp_end_CV = 3.5
    hold_CV = plan.final_plateau_mL / max(1e-6, cv_mL)

    include_flush = st.checkbox("Include 90% B flush", value=True, key=f"flush_{plan.mode}")
    flush_CV = 0.5 if include_flush else 0.0

    # Settings table (concise & legible)
    st.subheader("CombiFlash Method (enter these)")
    settings_df = pd.DataFrame(
        [
            ["Solvent A", "Hexane (PE)"],
            ["Solvent B", "Ethyl acetate (EA)"],
            ["Flow (mL/min)", f"{flow_disp}"],
            ["Equilibrate (mL)", f"{preeq_disp}"],
        ],
        columns=["Field", "Value"],
    )
    st.table(settings_df)

    # Gradient points (the instrument interpolates between points)
    points = []
    # P1: start
    points.append({"Point": "P1", "%B (EA)": 0.0, "CV target": 0.00, "Time (min)": 0.0})
    # P2: end of ramp
    t2 = ramp_end_CV * time_per_cv_min
    points.append({"Point": "P2", "%B (EA)": round(plan.final_pctEA, 1),
                   "CV target": round(ramp_end_CV, 2), "Time (min)": round(t2, 1)})
    # P3: end of hold
    cv3 = ramp_end_CV + hold_CV
    t3 = cv3 * time_per_cv_min
    points.append({"Point": "P3", "%B (EA)": round(plan.final_pctEA, 1),
                   "CV target": round(cv3, 2), "Time (min)": round(t3, 1)})
    # P4: optional flush
    if include_flush and flush_CV > 0:
        cv4 = cv3 + flush_CV
        t4 = cv4 * time_per_cv_min
        points.append({"Point": "P4", "%B (EA)": 90.0,
                       "CV target": round(cv4, 2), "Time (min)": round(t4, 1)})

    st.caption("Gradient points (the instrument interpolates between points)")
    st.dataframe(
        pd.DataFrame(points, columns=["Point", "%B (EA)", "CV target", "Time (min)"]),
        use_container_width=True,
        hide_index=True,
    )

# ===================== ROUTE: post_run =====================
if st.session_state["route"] == "post_run":
    render_post_run(
        sb=sb,
        default_vendor="Teledyne ISCO CombiFlash Rf+",
        default_cartridge=st.session_state.get("last_cartridge"),
        ref_plan=st.session_state.get("last_plan_dict"),
        raw_inputs=st.session_state.get("raw_inputs"),
        sample_id=st.session_state.get("last_sample_id"),
        sample_name=st.session_state.get("last_sample_name"),
        email_default=st.session_state.get("last_email"),
    )
    st.stop()

# ===================== HEADER & CREDIT =====================
st.markdown(
    """
**Built on:**
Hao Xu, Wenchao Wu, Yuntian Chen, Dongxiao Zhang, Fanyang Mo,  
*Explicit relation between thin film chromatography and column chromatography conditions from statistics and machine learning*,  
**Nature Communications** (2025) 16:832. DOI: 10.1038/s41467-025-56136-x
    """
)
st.title("🧪 SmartColumn")

# ===================== SIDEBAR =====================
with st.sidebar:
    if _sb_err:
        st.warning(f"Supabase not configured: {_sb_err}")

    st.header("Mode")
    ui_mode = st.selectbox(
        "Select system",
        options=["Autocolumn (CombiFlash Rf+)", "Hand column (glass)"],
        index=0
    )

    st.header("Inputs")
    rf = st.number_input("TLC Rf (product)", 0.0, 0.99, 0.30, 0.01, format="%.2f")
    tlc_pctEA = st.number_input("TLC %EA (PE/EA)", 0.0, 70.0, 20.0, 1.0, format="%.1f")
    mass_g = st.number_input("Sample mass (g)", 0.1, 1000.0, 5.0, 0.1, format="%.1f")
 
    autocolumn = ui_mode.startswith("Autocolumn")
    vendor_name = "Teledyne ISCO CombiFlash Rf+" if autocolumn else None
    cartridge_name = None
    silica_override_g = None
    preeq_CV_override = None

    if autocolumn:
        cart_names = [c["name"] for c in COMBIFLASH_CARTS]
        idx_default = 3  # default to 40 g
        cart_sel = st.selectbox("Cartridge", options=cart_names, index=idx_default)
        selected = next(c for c in COMBIFLASH_CARTS if c["name"] == cart_sel)
        cartridge_name = selected["name"]
        silica_override_g = float(selected["silica_g"])
        preeq_CV_override = 2.0  # vendor equilibration default
        st.session_state["last_cartridge"] = cartridge_name

    st.header("View")
    view = st.radio("View", ("Plan", "How it works"), index=0)

    st.markdown("---")
    if st.button("📥 Open Post-run Intake"):
        goto(
            "post_run",
            raw_inputs={
                "rf": rf, "tlc_pctEA": tlc_pctEA, "mass_g": mass_g,
                "silica_override_g": silica_override_g, "preeq_CV_override": preeq_CV_override,
            },
        )

    if st.button("🔄 Restart App"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        goto("plan")

# --- Motivation block shown on the home page before planning ---
if st.session_state.get("route") == "plan" and 'view' in locals() and view == "Plan":
    st.markdown("""
### Why help train this system?
Your help is needed to lock in a working method for completing the data chain from **TLC → column conditions**, by adding the column conditions to a practical plug-and-play column protocol. Ultimately this saves **time, money, and solvent waste** for organic chemistry labs.

**Authorship credit:** Every user who submits **clean post-run data** and a **SMILES** string for the target molecule will receive credit on an eventual publication, with **top contributors** added to the **main authorship list**.

Thanks for helping bring the dream of **fully automated synthesis** closer to reality!
""")


# ===================== AUX VIEW =====================
if view == "How it works":
    try:
        show_mathjax_html("SmartColumn_Exact.html", height=900)
    except Exception:
        st.info("Documentation view not found. Add SmartColumn_Exact.html to project root.")
    st.stop()

# ===================== MAIN: PLAN UI =====================
cfg = PlannerConfig()
btn = st.button("Plan Column", type="primary")

def render_mode(mode: str, autocolumn_flag: bool):
    plan = plan_column(
        rf=rf,
        tlc_pctEA=tlc_pctEA,
        mass_mg=mass_g * 1000.0,
        standard_ids_cm=STANDARD_GLASS_IDS_CM,
        cfg=cfg,
        mode=mode,
        silica_override_g=silica_override_g,
        preeq_CV_override=preeq_CV_override,
        vendor_name=vendor_name,
        cartridge_name=cartridge_name
    )

    st.markdown(f"### {mode_label(mode)}")
    highlight_block(plan, autocolumn=autocolumn_flag)

    if autocolumn_flag:
        # Show vendor-ready CombiFlash method
        render_combiflash_method(plan)
    else:
        # Hand-column: allow ID picker & show details/costs
        default_id = round(plan.glass_id_cm * 2) / 2
        sel_id = st.selectbox(
            "Select available column ID (cm) and recalc",
            options=[round(x, 1) for x in STANDARD_GLASS_IDS_CM],
            index=[round(x, 1) for x in STANDARD_GLASS_IDS_CM].index(round(default_id, 1)),
            key=f"sel_id_{mode}"
        )
        if abs(sel_id - plan.glass_id_cm) > 1e-6:
            plan = plan_column(
                rf=rf,
                tlc_pctEA=tlc_pctEA,
                mass_mg=mass_g * 1000.0,
                standard_ids_cm=STANDARD_GLASS_IDS_CM,
                cfg=cfg,
                mode=mode,
                override_id_cm=float(sel_id),
                vendor_name=vendor_name,
                cartridge_name=cartridge_name
            )
            highlight_block(plan, autocolumn=autocolumn_flag)

        with st.expander("Details & costs", expanded=True):
            details_tables(plan)

    return plan

if btn:
    hdr = f"Inputs → Rf: {round(rf, 2)}, TLC: {round(tlc_pctEA)}% EA, mass: {round(mass_g, 1)} g"
    if autocolumn:
        hdr += f"  |  System: CombiFlash Rf+ · {cartridge_name}"
    else:
        hdr += "  |  System: Hand column"
    st.subheader(hdr)

    c1, c2, c3 = st.columns(3, gap="large")
    with c1: plan_cons = render_mode("conservative", autocolumn)
    with c2: plan_std  = render_mode("standard", autocolumn)
    with c3: plan_eff  = render_mode("efficient", autocolumn)

    # Stash canonical STANDARD plan + raw inputs for post-run
    st.session_state["last_plan_dict"] = asdict(plan_std)
    st.session_state["raw_inputs"] = {
        "rf": rf, "tlc_pctEA": tlc_pctEA, "mass_g": mass_g,
        "silica_override_g": silica_override_g, "preeq_CV_override": preeq_CV_override,
    }
else:
    st.info("Set inputs, pick **CombiFlash Rf+** cartridge (or Hand column), then click **Plan Column**.")
