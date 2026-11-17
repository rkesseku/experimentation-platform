"""Streamlit readout dashboard.

Run:  streamlit run src/expplatform/dashboard.py

Uses simulated data by default (toggle an SRM bug to see the alarm), or upload a unit-level CSV
with ``unit_id, variant, ts, y, x_pre, ...`` columns.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from expplatform.demo import demo_config, demo_frame
from expplatform.report import build_readout


def _banner(out: dict) -> None:
    status = out["status"]
    msg = {
        "invalid_srm": "Invalid: sample-ratio mismatch. Do not read the effects.",
        "ship_candidate": "Primary metric significant (multiplicity-corrected), guardrails pass.",
        "hold_guardrail": "Primary significant but a guardrail is not non-inferior.",
        "inconclusive": "Inconclusive: no corrected-significant primary effect.",
    }[status]
    (st.error if status in ("invalid_srm", "hold_guardrail") else
     st.success if status == "ship_candidate" else st.info)(msg)


def main() -> None:
    st.set_page_config(page_title="Experiment readout", layout="wide")
    st.title("Experiment readout")
    with st.sidebar:
        src = st.radio("Data", ["Simulated demo", "Upload CSV"])
        if src == "Simulated demo":
            seed = st.number_input("Seed", 0, 10_000, 7)
            n = st.slider("Units", 5_000, 100_000, 40_000, step=5_000)
            srm = st.slider("SRM bug: fraction of treated units lost", 0.0, 0.1, 0.0, 0.005)
            df = demo_frame(int(seed), int(n), srm)
        else:
            up = st.file_uploader("Unit-level CSV", type="csv")
            if up is None:
                st.stop()
            df = pd.read_csv(up)
    cfg = demo_config()
    out = build_readout(df, cfg)
    _banner(out)

    c1, c2, c3 = st.columns(3)
    c1.metric("Units", f"{out['n_units']:,}")
    c2.metric("SRM p-value", f"{out['srm']['p_value']:.4f}",
              "MISMATCH" if out["srm"]["mismatch"] else "ok", delta_color="inverse")
    c3.metric("Assignment mismatches", out["assignment_check"]["mismatched"])

    st.subheader("Effects")
    rows = []
    for m in out["metrics"]:
        e = m["effect"]
        rows.append({"metric": m["name"], "role": m["role"], "effect": e["estimate"],
                     "ci_low": e["ci_low"], "ci_high": e["ci_high"],
                     "p_adjusted": m["p_adjusted"], "variance_reduction": m["variance_reduction"]})
    st.dataframe(pd.DataFrame(rows), width="stretch")

    st.subheader("Guardrails (non-inferiority)")
    st.dataframe(pd.DataFrame([{k: g[k] for k in ("name", "margin", "estimate",
                 "one_sided_bound", "p_value", "non_inferior")} for g in out["guardrails"]]),
                 width="stretch")

    seq = out["sequential"]
    if seq.get("enabled"):
        st.subheader("Anytime-valid confidence sequence (primary metric)")
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.fill_between(seq["n"], seq["cs_low"], seq["cs_high"], alpha=0.25, label="95% CS")
        ax.plot(seq["n"], seq["estimate"], lw=1.5, label="estimate")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("units observed")
        ax.set_ylabel("effect")
        ax.legend()
        st.pyplot(fig)
        st.caption(f"mSPRT rejects H0 at n = {seq['rejected_at_n']}" if seq["rejected_at_n"]
                   else "mSPRT has not rejected H0.")
        if seq["obf"]:
            st.write("O'Brien-Fleming checkpoints")
            st.dataframe(pd.DataFrame(seq["obf"]), width="stretch")

    if out.get("segments") and out["segments"]["segments"]:
        st.subheader("Segment effects (corrected)")
        st.dataframe(pd.DataFrame(out["segments"]["segments"])[
            ["segment", "estimate", "ci_low", "ci_high", "p_value", "p_adjusted", "significant"]],
            width="stretch")
        st.caption(f"Heterogeneity (Cochran Q) p = {out['segments']['q_p_value']:.4f}")


main()
