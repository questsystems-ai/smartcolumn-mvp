# SmartColumn MVP — Rf → Column Chromatography Planner

This is a **Streamlit** app that implements the MVP you requested:
- Input: **TLC Rf**, **TLC solvent ratio (PE/EA)**, and **sample mass (mg)**
- Output: **silica grams**, **multi-step gradient (PE/EA)** with volumes and plateau times,
  **predicted elution window**, **flow**, total solvent/time, and **green metrics**.

### What’s inside
- `app.py` — Streamlit UI
- `planner/equations.py` — Pluggable equations that map TLC **Rf** and **PE:EA** to mean CC **V_S**/**V_E**
  (Xu et al., Nature Communications 2025). **Coefficients are configurable** in `data/coeffs_default.json`.
- `planner/planner.py` — Gradient planning logic (ladder of %EA steps, plateau sizing, ramps)
- `planner/utils.py` — Helpers/validation
- `data/coeffs_default.json` — Default coefficients for the equations; you can edit these for lab calibration.
- `requirements.txt` — Dependencies
- `tests/demo_inputs.json` — Example input set for quick testing

> **Scope (MVP):** Normal-phase silica; **PE/EA only** (as in Xu et al.). Typical small-molecule organics that show clean TLC in PE/EA.
> Results are first-order/compound-agnostic; you can later calibrate a lab-specific scaling factor (epsilon) or refit coefficients.

### Quick start
```bash
# 1) Create & activate a virtual environment (optional)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the app
streamlit run app.py
```

Then open the URL Streamlit prints (e.g., http://localhost:8501).

### Notes
- The **Xu et al. equations** are implemented here in a pluggable way and **scaled to column size by grams silica**.
- Default coefficients were transcribed from the paper’s reported symbolic forms for the **PE/EA** system and are **approximate**;
  please calibrate in your lab by editing `data/coeffs_default.json` (or add a few local runs and we’ll auto-fit).
