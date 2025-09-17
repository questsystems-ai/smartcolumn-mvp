# post_run.py
from __future__ import annotations
import io, uuid, re
from typing import Optional, Dict, Any, TYPE_CHECKING
import pandas as pd
import streamlit as st

# ---------- Supabase (TYPE_CHECKING-safe) ----------
try:
    from supabase import create_client  # pip install supabase
except Exception:
    create_client = None  # type: ignore

if TYPE_CHECKING:
    from supabase import Client as SupabaseClient
else:
    from typing import Any
    SupabaseClient = Any  # type: ignore

def get_sb() -> Optional[SupabaseClient]:
    """Create a Supabase client from env vars; return None if not configured."""
    import os
    if create_client is None:
        return None
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
    if not url or not key:
        return None
    return create_client(url, key)

def upload_bytes(sb: SupabaseClient, bucket: str, path: str, data: bytes) -> str:
    """Upload bytes to Supabase Storage; returns path (throws on failure)."""
    f = io.BytesIO(data)
    try:
        sb.storage.from_(bucket).remove([path])  # overwrite if exists
    except Exception:
        pass
    sb.storage.from_(bucket).upload(path, f)
    return path

# ---------- Optional planner import (for on-the-fly plan) ----------
STANDARD_GLASS_IDS_CM = [x / 2 for x in range(1, 25)]  # 0.5..12.0
try:
    from planner import plan_column, PlannerConfig  # your existing module
except Exception:
    plan_column = None   # type: ignore
    PlannerConfig = None # type: ignore

# ---------- Parse vendor export & metrics ----------
def _read_text_table(file_bytes: bytes) -> pd.DataFrame:
    txt = file_bytes.decode("utf-8", errors="ignore")
    return pd.read_csv(io.StringIO(txt), sep=None, engine="python")

def parse_vendor_export(uploaded) -> Dict[str, Any]:
    name = uploaded.name
    raw = uploaded.getvalue()
    ext = (name.split(".")[-1] or "").lower()
    out = {"df": None, "raw_bytes": raw, "filetype": ext, "filename": name}
    if ext in {"csv", "txt"}:
        try:
            out["df"] = _read_text_table(raw)
            out["filetype"] = "csv" if ext == "csv" else "txt"
        except Exception as e:
            st.warning(f"Could not parse {ext.upper()} as table: {e}")
    elif ext == "pdf":
        out["filetype"] = "pdf"
    else:
        out["filetype"] = "other"
    return out

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Time": "time_s", "time": "time_s", "Time (s)": "time_s", "Seconds": "time_s",
        "UV": "uv", "Absorbance": "uv", "Signal": "uv", "UV1": "uv", "Detector": "uv",
        "B%": "pctB", "%B": "pctB", "EA%": "pctB", "%EA": "pctB",
        "Cumulative Volume (mL)": "cum_vol_mL", "Cumulative Volume": "cum_vol_mL",
        "Volume (mL)": "cum_vol_mL", "Total Volume (mL)": "cum_vol_mL",
        "Fraction": "frac_idx", "Fraction #": "frac_idx", "Frac": "frac_idx",
    }
    for c in list(df.columns):
        if c in rename_map:
            df = df.rename(columns={c: rename_map[c]})
    return df

def compute_vs_ve_from_trace(df: pd.DataFrame, final_pctEA: Optional[float]) -> Dict[str, Optional[float]]:
    out = {"VS_obs_mL": None, "VE_obs_mL": None, "total_solvent_mL": None}
    if "cum_vol_mL" in df.columns:
        out["total_solvent_mL"] = float(df["cum_vol_mL"].max())

    df2 = df.copy()
    if "pctB" in df2.columns and final_pctEA is not None:
        tol = 1.0
        mask = df2["pctB"].round(1).between(final_pctEA - tol, final_pctEA + tol)
        if mask.sum() >= 10:
            df2 = df2[mask]

    if {"uv", "cum_vol_mL"}.issubset(df2.columns) and len(df2):
        uv = df2["uv"].astype(float).values
        vmax = float(uv.max())
        if vmax > 0:
            thr = 0.10 * vmax
            vols = df2["cum_vol_mL"].astype(float).values
            vs_idx = next((i for i, v in enumerate(uv) if v >= thr), None)
            ve_idx = None
            for j, v in enumerate(reversed(uv)):
                if v >= thr:
                    ve_idx = len(uv) - 1 - j
                    break
            if vs_idx is not None:
                out["VS_obs_mL"] = float(vols[vs_idx])
            if ve_idx is not None:
                out["VE_obs_mL"] = float(vols[ve_idx])
    return out

# ---------- UI ----------
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def _ensure_plan_row(
    sb: SupabaseClient,
    ref_plan: Optional[Dict[str, Any]],
    raw_inputs: Optional[Dict[str, Any]],
    sample_id: Optional[str],
    vendor: Optional[str],
    cartridge: Optional[str],
) -> str:
    # already saved?
    if ref_plan and ref_plan.get("plan_id"):
        return ref_plan["plan_id"]

    row: Dict[str, Any] = {"sample_id": sample_id, "vendor": vendor, "cartridge": cartridge}

    if ref_plan:
        row.update({
            "rf": raw_inputs.get("rf") if raw_inputs else None,
            "tlc_pct_ea": raw_inputs.get("tlc_pctEA") if raw_inputs else None,
            "mass_g": raw_inputs.get("mass_g") if raw_inputs else None,
            "mode": ref_plan.get("mode", "standard"),
            "silica_g": ref_plan.get("silica_g"),
            "glass_id_cm": ref_plan.get("glass_id_cm"),
            "bed_height_cm": ref_plan.get("bed_height_cm"),
            "cv_ml": ref_plan.get("column_volume_mL"),
            "final_pct_ea": ref_plan.get("final_pctEA"),
            "predicted_vs_ml": (ref_plan.get("predicted_elution_mL") or [None, None])[0],
            "predicted_ve_ml": (ref_plan.get("predicted_elution_mL") or [None, None])[1],
            "total_solvent_ml": ref_plan.get("total_solvent_mL"),
            "total_time_min": ref_plan.get("total_time_min"),
        })
    elif plan_column and raw_inputs:
        cfg = PlannerConfig() if PlannerConfig else None
        plan = plan_column(
            rf=float(raw_inputs["rf"]),
            tlc_pctEA=float(raw_inputs["tlc_pctEA"]),
            mass_mg=float(raw_inputs["mass_g"]) * 1000.0,
            standard_ids_cm=STANDARD_GLASS_IDS_CM,
            cfg=cfg,
            mode="standard",
            silica_override_g=raw_inputs.get("silica_override_g"),
            preeq_CV_override=raw_inputs.get("preeq_CV_override"),
            vendor_name=vendor,
            cartridge_name=cartridge,
        )
        row.update({
            "rf": raw_inputs["rf"],
            "tlc_pct_ea": raw_inputs["tlc_pctEA"],
            "mass_g": raw_inputs["mass_g"],
            "mode": plan.mode,
            "silica_g": plan.silica_g,
            "glass_id_cm": plan.glass_id_cm,
            "bed_height_cm": plan.bed_height_cm,
            "cv_ml": plan.column_volume_mL,
            "final_pct_ea": plan.final_pctEA,
            "predicted_vs_ml": plan.predicted_elution_mL[0],
            "predicted_ve_ml": plan.predicted_elution_mL[1],
            "total_solvent_ml": plan.total_solvent_mL,
            "total_time_min": plan.total_time_min,
        })
    else:
        if raw_inputs:
            row.update({
                "rf": raw_inputs.get("rf"),
                "tlc_pct_ea": raw_inputs.get("tlc_pctEA"),
                "mass_g": raw_inputs.get("mass_g"),
                "mode": "standard",
            })

    res = sb.table("plans").insert(row).execute()
    return res.data[0]["id"]

def _topbar_restart_only():
    right = st.columns([7, 3])[1]
    if right.button("🔄 Restart App", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

def render_post_run(
    sb: Optional[SupabaseClient] = None,
    default_vendor: str = "Teledyne ISCO CombiFlash Rf+",
    default_cartridge: Optional[str] = None,
    ref_plan: Optional[Dict[str, Any]] = None,
    raw_inputs: Optional[Dict[str, Any]] = None,   # rf, tlc_pctEA, mass_g, overrides...
    sample_id: Optional[str] = None,
    sample_name: Optional[str] = None,
    smiles: Optional[str] = None,
    email_default: Optional[str] = None,
):
    """Post-run intake. On Submit: saves sample, plan (if needed), uploads, inserts run."""
    _topbar_restart_only()
    st.title("📥 Post-run intake")

    with st.expander("Context", expanded=True):
        top = st.columns(4)
        vendor = top[0].text_input("Vendor / System", value=default_vendor or "")
        cartridge = top[1].text_input("Cartridge", value=default_cartridge or "")
        email = top[2].text_input("Your email (required)", value=email_default or "", placeholder="name@lab.org")
        top[3].markdown(f"**SMILES:** `{smiles}`" if smiles else "&nbsp;")

        st.session_state["_post_vendor"] = vendor
        st.session_state["_post_cartridge"] = cartridge
        st.session_state["_post_email"] = email

        if ref_plan:
            m = st.columns(3)
            m[0].metric("Planned final %EA", f"{round(ref_plan.get('final_pctEA', 0))}%")
            m[1].metric("Planned total solvent", f"{round(ref_plan.get('total_solvent_mL', 0))} mL")
            pe = (ref_plan.get('predicted_elution_mL') or [None, None])[1] or 0
            m[2].metric("Planned V̄E", f"{round(pe)} mL")

    st.subheader("Upload files")
    run_file = st.file_uploader("Vendor run export (.csv / .txt / .pdf)", type=["csv", "txt", "pdf"])
    tlc_photo = st.file_uploader("TLC photo (PNG/JPG)", type=["png", "jpg", "jpeg"])

    df_preview = None
    obs = {"VS_obs_mL": None, "VE_obs_mL": None, "total_solvent_mL": None}
    if run_file is not None:
        parsed = parse_vendor_export(run_file)
        if parsed["df"] is not None:
            df = normalize_columns(parsed["df"])
            df_preview = df.head(30)
            st.caption("Parsed table (preview)")
            st.dataframe(df_preview, use_container_width=True)
            final_pctEA_hint = (ref_plan or {}).get("final_pctEA")
            obs = compute_vs_ve_from_trace(df, final_pctEA=final_pctEA_hint)
        else:
            st.info("Stored file for record (PDF/other); VS/VE will need manual entry.")

    st.subheader("Observed metrics")
    c1, c2, c3, c4 = st.columns(4)
    vs_ml = c1.number_input("Observed V̄S (mL)", value=float(obs["VS_obs_mL"] or 0.0), step=1.0)
    ve_ml = c2.number_input("Observed V̄E (mL)", value=float(obs["VE_obs_mL"] or 0.0), step=1.0)
    total_ml = c3.number_input("Total solvent (mL)", value=float(obs["total_solvent_mL"] or 0.0), step=10.0)

    delta_ml = None
    if ref_plan and ref_plan.get("total_solvent_mL"):
        delta_ml = total_ml - float(ref_plan["total_solvent_mL"])
        c4.metric("Δ solvent vs plan", f"{int(delta_ml)} mL",
                  f"{(100.0*delta_ml/max(1.0, ref_plan['total_solvent_mL'])):+.0f}%")

    st.subheader("Outcome & notes")
    c1, c2, c3 = st.columns(3)
    outcome = c1.radio("Outcome", ["Good", "Acceptable", "Failed"], index=1, horizontal=True)
    purity = c2.number_input("Observed purity (%)", 0.0, 100.0, value=0.0, step=1.0)
    yield_pct = c3.number_input("Yield (%)", 0.0, 100.0, value=0.0, step=1.0)

    st.subheader("Loading details")
    mode = st.selectbox("Loading mode", ["Liquid (in solvent)", "Dryload"])
    loading_pctEA = None
    loading_vol_mL = None
    dryload_silica_g = None
    if mode == "Liquid (in solvent)":
        c1, c2 = st.columns(2)
        loading_pctEA = c1.number_input("Loading solvent %EA", 0.0, 100.0, value=0.0, step=1.0)
        loading_vol_mL = c2.number_input("Loading solvent volume (mL)", 0.0, 1000.0, value=0.0, step=1.0)
    else:
        dryload_silica_g = st.number_input("Dryload silica (g)", 0.0, 1000.0, value=0.0, step=1.0)

    notes = st.text_area("Notes (optional)", "")

    st.markdown("---")
    save = st.button("Submit (save plan + run to Supabase)")

    if save:
        # email required
        email_val = (st.session_state.get("_post_email") or "").strip()
        if not _EMAIL_RE.match(email_val):
            st.error("Please enter a valid email address (required).")
            st.stop()

        if sb is None:
            st.error("Supabase is not configured (missing env or client).")
            st.stop()

        # ensure sample (optional in your current schema; if you want to persist sample_name/smiles, insert in app flow)
        sid = sample_id  # keep as provided (or extend to insert if you want)

        # ensure plan (insert now if needed)
        vendor = st.session_state.get("_post_vendor")
        cartridge = st.session_state.get("_post_cartridge")
        plan_id = _ensure_plan_row(
            sb, ref_plan=ref_plan, raw_inputs=raw_inputs,
            sample_id=sid, vendor=vendor, cartridge=cartridge
        )

        # upload files
        run_uuid = str(uuid.uuid4())
        export_path = None
        tlc_path = None
        if run_file is not None:
            export_path = f"{run_uuid}/{run_file.name}"
            upload_bytes(sb, "vendor_exports", export_path, run_file.getvalue())
        if tlc_photo is not None:
            tlc_path = f"{run_uuid}/{tlc_photo.name}"
            upload_bytes(sb, "tlc_images", tlc_path, tlc_photo.getvalue())

        # insert run row
        row_run = {
            "plan_id": plan_id,
            "vendor": vendor,
            "cartridge": cartridge,
            "user_email": email_val,
            "observed_vs_ml": vs_ml or None,
            "observed_ve_ml": ve_ml or None,
            "total_solvent_ml": total_ml or None,
            "delta_solvent_ml": (delta_ml if delta_ml is not None else None),
            "outcome": outcome,
            "user_notes": notes,
            "export_path": export_path,
            "tlc_image_path": tlc_path,
            "observed_purity_pct": purity or None,
            "observed_yield_pct": yield_pct or None,
            "loading_mode": ("liquid" if mode.startswith("Liquid") else "dryload"),
            "loading_pct_ea": loading_pctEA,
            "loading_vol_ml": loading_vol_mL,
            "dryload_silica_g": dryload_silica_g,
            "sample_id": sid,
            "smiles": smiles,
        }
        try:
            res = sb.table("runs").insert(row_run).execute()
            run_id = res.data[0]["id"]
            st.success(f"Saved plan_id={plan_id} and run_id={run_id}")
        except Exception as e:
            st.error(f"Run insert failed: {e}")
