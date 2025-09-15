# app.py
import streamlit as st
import pandas as pd

from planner.planner import plan_column, PlannerConfig

# ---- economics (placeholders) ----
PRICE_PER_24L_USD  = 120.00    # solvent case
DISPOSAL_PER_L_USD = 6.00      # non-halogenated
SILICA_COST_PER_KG = 100.00    # loose silica

# Standard glass IDs we’ll try
STANDARD_GLASS_IDS_CM = [0,8, 10, 13, 15, 20, 25, 30, 40, 50]  # in mm for readability? We'll keep cm integers.
# Use cm values:
STANDARD_GLASS_IDS_CM = [0.8, 1.0, 1.3, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

# Biotage-style cartridges (placeholder)
BIOTAGE_CARTS = [
    (10,  "Sfär 10 g",   35),
    (25,  "Sfär 25 g",   49),
    (50,  "Sfär 50 g",   69),
    (100, "Sfär 100 g",  99),
    (185, "Sfär 185 g", 149),
    (340, "Sfär 340 g", 199),
    (500, "Sfär 500 g", 259),
]

def cart_for_silica(silica_g: float):
    for g, name, price in BIOTAGE_CARTS:
        if silica_g <= g:
            return name, price
    g, name, price = BIOTAGE_CARTS[-1]
    return name, price

def mode_label_emoji(mode: str) -> tuple[str, str]:
    return (
        ("Conservative", "🛡️") if mode == "conservative" else
        ("Efficient", "⚡")     if mode == "efficient"     else
        ("Standard", "⚖️")
    )

st.set_page_config(page_title="SmartColumn — Plan Column", page_icon="🧪", layout="wide")
st.title("🧪 SmartColumn — Plan Column (Rf → 12 cm Bed ±2 cm + Column ID + Summary Gradient)")
st.caption("Normal-phase silica • PE/EA • CV = porosity × bed volume; Xu-style oracle picks final %EA; STANDARD uses 2 CV increments (🛡️=3 CV, ⚡=1.5 CV)")

st.markdown(
    "#### Choosing a mode from impurity separation\n"
    "- If nearest impurity **ΔRf ≤ 0.10** → **Conservative** (tighter bands / slower gradient)\n"
    "- If **0.10 < ΔRf ≤ 0.30** → **Standard**\n"
    "- If **ΔRf > 0.30** → **Efficient** (clean/easy systems)\n"
)

with st.sidebar:
    st.header("Inputs")
    rf = st.number_input("TLC Rf (product)", min_value=0.0, max_value=0.99, value=0.30, step=0.01, format="%.2f")
    tlc_pctEA = st.number_input("TLC %EA (PE/EA)", min_value=0.0, max_value=70.0, value=20.0, step=1.0, format="%.1f")
    mass_g = st.number_input("Sample mass (g)", min_value=0.1, value=0.1, step=0.1, format="%.1f")

cfg = PlannerConfig()
btn = st.button("Plan Column", type="primary")

def render_mode(mode: str, col):
    plan = plan_column(
        rf=rf,
        tlc_pctEA=tlc_pctEA,
        mass_mg=mass_g * 1000.0,
        standard_ids_cm=STANDARD_GLASS_IDS_CM,
        cfg=cfg,
        mode=mode
    )
    label, emoji = mode_label_emoji(mode)
    col.markdown(f"### {emoji} {label}")

    # Costs
    solvent_cost  = round((plan.total_solvent_mL / 24000.0) * PRICE_PER_24L_USD)
    disposal_cost = round((plan.total_solvent_mL / 1000.0) * DISPOSAL_PER_L_USD)
    silica_cost   = round((plan.silica_g / 1000.0) * SILICA_COST_PER_KG)
    cart_name, cart_price = cart_for_silica(plan.silica_g)

    # ---- Method summary (all integers) ----
    summary_rows = [
        ("TLC system (%EA)",            f"{round(plan.tlc_pctEA)}"),
        ("Silica (g)",                  f"{round(plan.silica_g)}"),
        ("Bed height (cm)",             f"{round(plan.bed_height_cm)}"),
        ("Column ID (cm)",              f"{round(plan.glass_id_cm)}"),
        ("Packed bed volume (mL)",      f"{round(plan.packed_volume_mL)}"),
        ("Column Volume, CV (mL)",      f"{round(plan.column_volume_mL)}"),
        ("Pre-equilibrate vol (mL)",    f"{round(plan.pre_equilibrate_mL)}"),
        ("Final isocratic (%EA)",       f"{round(plan.final_pctEA)}"),
        ("Increment size (%EA)",        f"{round(plan.increment_size_pctEA)}"),
        ("Increment volume (mL)",       f"{round(plan.increment_volume_mL)}"),
        ("Final plateau volume (mL)",   f"{round(plan.final_plateau_mL)}"),
        ("Predicted V̄S at TLC (mL)",   f"{round(plan.predicted_elution_tlc_mL[0])}"),
        ("Predicted V̄E at TLC (mL)",   f"{round(plan.predicted_elution_tlc_mL[1])}"),
        ("Predicted V̄S final (mL)",    f"{round(plan.predicted_elution_mL[0])}"),
        ("Predicted V̄E final (mL)",    f"{round(plan.predicted_elution_mL[1])}"),
        ("Total solvent (mL)",          f"{round(plan.total_solvent_mL)}"),
        ("Total time (min)",            f"{round(plan.total_time_min)}"),
    ]
    col.table(pd.DataFrame(summary_rows, columns=["Parameter", "Value"]))

    # ---- Cost table (integers only) ----
    cost_rows = [
        ("Loose silica (g)",            f"{round(plan.silica_g)}"),
        ("Loose silica cost (USD)",     f"{silica_cost}"),
        ("Pre-packed cartridge",        cart_name),
        ("Cartridge cost (USD)",        f"{cart_price}"),
        ("Solvent (mL)",                f"{round(plan.total_solvent_mL)}"),
        ("Solvent purchase cost (USD)", f"{solvent_cost}"),
        ("Disposal cost (USD)",         f"{disposal_cost}"),
    ]
    col.table(pd.DataFrame(cost_rows, columns=["Cost Item", "Value"]))

if btn:
    st.subheader(f"Inputs → Rf: **{round(rf,2)}**, TLC: **{round(tlc_pctEA)}% EA**, mass: **{round(mass_g,1)} g**")
    c1, c2, c3 = st.columns(3, gap="large")
    render_mode("conservative", c1)
    render_mode("standard",     c2)
    render_mode("efficient",    c3)

    st.divider()
    st.markdown("**Notes**")
    st.markdown(
        "- **CV** = porosity × bed volume; with ε≈0.4, a 12 cm × 1.3 cm bed is ≈16 mL packed, **CV ≈ 6 mL**.\n"
        "- **Silica grams** come from the load rule (x-ratio); Xu et al. used fixed columns and provide the **Rf + PE:EA → V̄S,V̄E** mapping (not silica sizing).\n"
        "- **STANDARD** uses **2 CV** for pre-eq & increments (🛡️ 3 CV, ⚡ 1.5 CV). **Final plateau** is **3 × increment**.\n"
        "- We keep the bed near **12 cm**; if a change > ±2 cm would be needed, we **choose a wider/narrower standard ID**."
    )
else:
    st.info("Enter Rf, TLC %EA, and mass (g), then click **Plan Column** to see three mode summaries and costs.")
