#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMSOL FFT Analyzer — Streamlit Web Interface
===============================================

Interactive browser-based UI for FFT analysis of COMSOL
time-domain pressure exports.  Re-uses fft_core.py for all
analysis logic.

Usage
-----
    py -m streamlit run app.py
"""

import io
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

import streamlit as st
from fft_core import (
    read_comsol_csv, detect_columns, split_by_parameter,
    check_time_uniformity, baseline_correct, compute_fft,
    normalize_spectrum, find_dominant_frequency, find_bw50,
    create_time_domain_figure, create_fft_spectrum_figure,
    _COLORS,
)

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="COMSOL FFT Analyzer",
    page_icon="\U0001f4ca",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-title{font-size:2.4rem;font-weight:700;
    background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    margin-bottom:.2rem}
  .sub-title{color:#999;font-size:1.05rem;margin-bottom:1.6rem}
  div[data-testid="stMetric"]{
    background:linear-gradient(135deg,#1a1a2e,#16213e);
    border:1px solid rgba(102,126,234,.3);border-radius:10px;
    padding:12px 16px}
  .stDownloadButton>button{width:100%}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">\U0001f4ca COMSOL FFT Analyzer</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-title">FFT-based frequency-domain analysis '
            'of COMSOL time-domain pressure exports</p>',
            unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────
def _fig_to_png(fig, dpi=300):
    """Return PNG bytes of a matplotlib figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def _metrics_to_csv(metrics):
    """Build metrics CSV string from a list of result dicts."""
    hdr = ("Point,Param_Name,Param_Value,Dominant_Frequency_MHz,"
           "Half_Max_Bandwidth_MHz,BW_F_Low_MHz,BW_F_High_MHz,"
           "Non_DC_Max_Amplitude,Freq_Resolution_MHz")
    rows = [hdr]
    for r in metrics:
        rows.append(
            f"{r['point']},{r.get('param_name','')},{r.get('param_value','')},"
            f"{r['dominant_freq']:.6f},{r['bandwidth']:.6f},"
            f"{r['bw_f_low']:.6f},{r['bw_f_high']:.6f},"
            f"{r['non_dc_max']:.10e},{r['freq_resolution']:.6f}")
    return "\n".join(rows)


def _spectra_to_csv(spectra, freq_max=500.0):
    """Build spectra CSV string."""
    header = ["Frequency_MHz"]
    cols, freq_ref = {}, None
    for sd in spectra:
        name = sd["label"]
        header.append(name)
        mask = sd["freq_mhz"] <= freq_max
        cols[name] = sd["normalized"][mask]
        if freq_ref is None:
            freq_ref = sd["freq_mhz"][mask]
    lines = [",".join(header)]
    for i in range(len(freq_ref)):
        row = [f"{freq_ref[i]:.6f}"]
        for n in header[1:]:
            row.append(f"{cols[n][i]:.10e}" if i < len(cols[n]) else "")
        lines.append(",".join(row))
    return "\n".join(lines)


def _report_text(metrics, cfg):
    """Build analysis report string."""
    bl = cfg.get("baseline_end_ns", 10.0)
    lines = [
        "=" * 70,
        "COMSOL FFT Analysis Report",
        "=" * 70,
        f"\nDate generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "Tool: COMSOL Photoacoustic FFT Analyzer (Streamlit UI)\n",
        "ANALYSIS SETTINGS", "-" * 40,
    ]
    for k, v in cfg.items():
        lines.append(f"  {k}: {v}")
    lines += [
        "", "FFT METHOD", "-" * 40,
        f"  Baseline correction: mean subtraction for t <= {bl} ns",
        "  Window: Hann (numpy.hanning)",
        "  FFT: numpy.fft.rfft (real FFT)",
        "  Normalization: non-DC maximum amplitude",
        "", "RESULTS", "-" * 40,
        f"{'Point':<20s} {'Param':<12s} {'f_dom (MHz)':>12s} "
        f"{'BW50 (MHz)':>12s} {'BW range (MHz)':>24s}",
    ]
    for r in metrics:
        pv = r.get("param_value", "-")
        lines.append(
            f"{r['point']:<20s} {str(pv):<12s} "
            f"{r['dominant_freq']:>12.3f} {r['bandwidth']:>12.3f} "
            f"[{r['bw_f_low']:>8.3f} -- {r['bw_f_high']:>8.3f}]")
    lines.append("")
    return "\n".join(lines)


# ── File upload ───────────────────────────────────────────────────
uploaded = st.file_uploader(
    "\U0001f4c1 Upload COMSOL CSV file", type=["csv"],
    help="COMSOL-exported CSV with time-domain pressure data. "
         "Lines starting with '%' are treated as metadata.")

if uploaded is None:
    st.info("Upload a COMSOL CSV file to begin.")
    st.stop()

# Save to temp file so fft_core.read_comsol_csv can read it
with tempfile.NamedTemporaryFile(delete=False, suffix=".csv",
                                 mode="wb") as tmp:
    tmp.write(uploaded.getvalue())
    _tmp_path = tmp.name

try:
    df = read_comsol_csv(_tmp_path)
except Exception as exc:
    st.error(f"Failed to parse CSV: {exc}")
    st.stop()

cols = list(df.columns)
st.success(f"\u2705 Parsed **{len(df)}** rows, **{len(cols)}** columns")

with st.expander("Preview raw data", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)

# ── Settings ──────────────────────────────────────────────────────
st.markdown("### \u2699\ufe0f Analysis Settings")
c1, c2 = st.columns(2)

with c1:
    point_name = st.text_input("Point name", value="Detector P3")
    param_opts = ["None (single signal)"] + cols
    param_sel = st.selectbox("Parameter column", param_opts)
    param_column = None if param_sel.startswith("None") else param_sel

    _ti = next((i for i, c in enumerate(cols) if "time" in c.lower()), 0)
    time_column = st.selectbox("Time column", cols, index=_ti)

    _si = next((i for i, c in enumerate(cols)
                if "pressure" in c.lower() or "photoacoustic" in c.lower()),
               min(len(cols) - 1, 2))
    signal_column = st.selectbox("Signal / pressure column", cols, index=_si)

with c2:
    baseline_end = st.number_input("Baseline end (ns)", value=10.0,
                                   min_value=0.0, step=1.0)
    freq_max = st.number_input("Plot freq max (MHz)", value=150.0,
                               min_value=1.0, step=10.0)
    export_freq_max = st.number_input("Export freq max (MHz)", value=500.0,
                                      min_value=1.0, step=50.0)
    dpi = int(st.number_input("Figure DPI", value=300,
                              min_value=72, max_value=600, step=50))

st.divider()

# ── Run analysis ──────────────────────────────────────────────────
if st.button("\U0001f680 Run FFT Analysis", type="primary",
             use_container_width=True):
    with st.spinner("Running FFT analysis\u2026"):
        col_map = {"time": time_column, "signal": signal_column,
                   "param": param_column}
        data_dict = split_by_parameter(df, col_map)
        p_name = param_column or "param"
        safe = point_name.replace(" ", "_")

        fft_results, metrics, spectra = {}, [], []
        for pval, (t_ns, sig) in sorted(
                data_dict.items(), key=lambda x: (x[0] is None, x[0])):
            check_time_uniformity(t_ns)
            corr, _ = baseline_correct(t_ns, sig, baseline_end)
            freq, amp = compute_fft(t_ns, corr)
            norm, ndc = normalize_spectrum(amp)
            dom_f, pk = find_dominant_frequency(freq, amp)
            bw, fl, fh = find_bw50(freq, norm, pk)
            df_r = freq[1] - freq[0]

            fft_results[pval] = dict(
                freq_mhz=freq, amplitude=amp, normalized=norm,
                non_dc_max=ndc, dominant_freq=dom_f, peak_idx=pk,
                bandwidth=bw, bw_f_low=fl, bw_f_high=fh)
            metrics.append(dict(
                point=point_name, param_name=p_name,
                param_value=pval, dominant_freq=dom_f, bandwidth=bw,
                bw_f_low=fl, bw_f_high=fh, non_dc_max=ndc,
                freq_resolution=df_r))
            lbl = (f"{safe}_{p_name}{pval}_normalized"
                   if pval is not None else f"{safe}_normalized")
            spectra.append(dict(label=lbl, freq_mhz=freq, normalized=norm))

        fig_t = create_time_domain_figure(data_dict, point_name, p_name)
        fig_f = create_fft_spectrum_figure(fft_results, point_name,
                                           freq_max, p_name)
        cfg = dict(point_name=point_name, baseline_end_ns=baseline_end,
                   freq_max_mhz=freq_max, export_freq_max_mhz=export_freq_max,
                   figure_dpi=dpi)

        # Store everything in session state so results survive reruns
        st.session_state.res = dict(
            metrics=metrics, spectra=spectra,
            fig_t_png=_fig_to_png(fig_t, dpi),
            fig_f_png=_fig_to_png(fig_f, dpi),
            fig_t=fig_t, fig_f=fig_f,
            cfg=cfg, export_freq_max=export_freq_max,
            safe=safe)
        plt.close(fig_t)
        plt.close(fig_f)

# ── Display results ───────────────────────────────────────────────
if "res" not in st.session_state:
    st.stop()

R = st.session_state.res

st.markdown("---")
st.markdown("## \U0001f4c8 Results")

# Key metrics
mcols = st.columns(len(R["metrics"]))
for i, m in enumerate(R["metrics"]):
    with mcols[i]:
        pv = m.get("param_value", "")
        st.metric(f"{m['point']} ({pv})",
                  f"{m['dominant_freq']:.3f} MHz",
                  f"BW50 = {m['bandwidth']:.3f} MHz")

# Tabs for plots / table / report
tab_td, tab_fft, tab_tbl, tab_rpt = st.tabs(
    ["\U0001f4c9 Time Domain", "\U0001f4ca FFT Spectrum",
     "\U0001f4cb Metrics Table", "\U0001f4dd Report"])

with tab_td:
    st.pyplot(R["fig_t"])
    st.download_button("\u2b07\ufe0f Download Time-Domain PNG",
                       R["fig_t_png"],
                       file_name=f"{R['safe']}_raw_time_domain_pressure.png",
                       mime="image/png")

with tab_fft:
    st.pyplot(R["fig_f"])
    st.download_button("\u2b07\ufe0f Download FFT Spectrum PNG",
                       R["fig_f_png"],
                       file_name=f"{R['safe']}_normalized_fft_spectrum.png",
                       mime="image/png")

with tab_tbl:
    tbl = pd.DataFrame(R["metrics"])
    display_cols = ["point", "param_name", "param_value",
                    "dominant_freq", "bandwidth",
                    "bw_f_low", "bw_f_high", "freq_resolution"]
    st.dataframe(tbl[display_cols], use_container_width=True,
                 hide_index=True)
    csv_str = _metrics_to_csv(R["metrics"])
    st.download_button("\u2b07\ufe0f Download Metrics CSV",
                       csv_str, file_name="fft_metrics.csv",
                       mime="text/csv")
    spec_str = _spectra_to_csv(R["spectra"], R["export_freq_max"])
    st.download_button("\u2b07\ufe0f Download Spectra CSV",
                       spec_str, file_name="fft_spectra.csv",
                       mime="text/csv")

with tab_rpt:
    report = _report_text(R["metrics"], R["cfg"])
    st.code(report, language=None)
    st.download_button("\u2b07\ufe0f Download Report TXT",
                       report, file_name="analysis_report.txt",
                       mime="text/plain")
