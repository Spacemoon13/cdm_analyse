import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io
import requests
import base64
import datetime

# ================================================================
# Streamlit Grund-Setup
# ================================================================
st.set_page_config(
    page_title="ACG CDM Delta Dashboard",
    page_icon="✈️",
    layout="wide",
)

# ================================================================
# Passwortschutz
# ================================================================
def check_password():
    def password_entered():
        pwd = st.session_state.get("password_input", "")
        correct_pwd = st.secrets["auth"]["password"]
        if pwd == correct_pwd:
            st.session_state["password_correct"] = True
            del st.session_state["password_input"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.markdown(
        """
        <div style="padding:2rem 1rem 0.5rem 1rem; text-align:center;">
          <h1 style="color:#003DA5; margin-bottom:0.2rem;"> CDM Dashboard – Login </h1>
          <p style="color:#555; font-size:0.95rem;"> Bitte Passwort eingeben, um das Dashboard zu öffnen. </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.text_input(
        "Passwort",
        type="password",
        key="password_input",
        on_change=password_entered,
    )

    if st.session_state.get("password_correct") is False:
        st.error("Falsches Passwort.")

    return False


# ================================================================
# Einstellungen / Defaults
# ================================================================
SHEET_NAME = "Deltas"
BIN_SIZE = 5
TIME_MIN = 0
DELTA_LIMIT = 120
MIN_COUNT = 200

AIRLINE_CATEGORIES = {
    "DLH Group": ["AUA", "SWR", "EWG", "DLH", "BEL", "CLH", "DLA", "OCN"],
    "Low Cost Carrier": ["RYR", "WZZ", "WMT", "EZY", "EZS", "EJU", "VLG", "EXS", "TRA", "TVF", "CAI", "CXI", "SXS", "PGT", "TKJ"],
    "Long Haul": ["UAE", "QTR", "ETD", "ACA", "EVA", "ETH", "CHH", "CAL", "KAL", "JAL", "AIC", "ABY", "CCA"],
    "Biz Jets": ["VJT", "NJE", "AWH", "GDK", "AOJ", "LDX", "VCJ", "JFL", "TJS", "PTN", "GCK", "IFA", "IJM", "UAG", "FSF", "PAV", "PVD", "BTX", "TOY", "MPC", "OEE", "OEH", "OEF", "OEI"],
}

CATEGORIES_OF_INTEREST = ["DLH Group", "Low Cost Carrier", "Long Haul", "Biz Jets"]
RUNWAYS_OF_INTEREST = ["11", "16", "29", "34"]

sns.set(style="whitegrid")

colors = {
    "etot": "#003DA5",
    "ctot": "#FF6900",
    "atc": "#6E6E6E",
}


def smooth(series, window=3):
    return series.rolling(window=window, center=True, min_periods=1).mean()


# ================================================================
# SPALTENNAMEN
# ================================================================
COL_MIN_TO_ATOT = "Min bis ATOT"

COL_ETOT = "DeltaAbs - ETOT (min)"
COL_CTOT = "DeltaAbs - CTOT (min)"
COL_ATC = "DeltaAbs - ATC TTOT (min)"

# SIGNED (für Histogramm)
COL_ETOT_S = "DeltaSigned - ETOT (min)"
COL_CTOT_S = "DeltaSigned - CTOT (min)"
COL_ATC_S = "DeltaSigned - ATC TTOT (min)"

NUMERIC_COLS = [COL_MIN_TO_ATOT, COL_ETOT, COL_CTOT, COL_ATC]


# ================================================================
# Daten laden (GitHub raw URL kommt aus secrets)
# ================================================================
@st.cache_data
def load_data():
    url = st.secrets["file_links"]["xlsm_url"]

    try:
        with requests.Session() as s:
            resp = s.get(url, timeout=30)
            resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Fehler beim Laden der Daten: {e}")

    df = pd.read_excel(
        io.BytesIO(resp.content),
        sheet_name=SHEET_NAME,
        engine="openpyxl",
    )

    required_cols = NUMERIC_COLS + ["Runway", "Airline", COL_ETOT_S, COL_CTOT_S, COL_ATC_S]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Fehlende Spalten in Excel: {missing}")

    for col in (NUMERIC_COLS + [COL_ETOT_S, COL_CTOT_S, COL_ATC_S]):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[COL_MIN_TO_ATOT])

    df["bin"] = (df[COL_MIN_TO_ATOT] / BIN_SIZE).astype(int) * BIN_SIZE
    df["Runway"] = df["Runway"].astype(str).str.strip()
    df["Airline"] = df["Airline"].astype(str).str.strip()

    airline_map = {}
    for cat, codes in AIRLINE_CATEGORIES.items():
        for c in codes:
            airline_map[c] = cat
    df["AirlineCategory"] = df["Airline"].map(airline_map).fillna("Other")

    return df


def compute_stats(data, col, limit):
    mask = data[col].notna() & data[col].between(-limit, limit)
    sub = data[mask]
    return sub.groupby("bin")[col].agg(mean="mean", count="count").sort_index()


def percent_within_window(df, bins, col, window, limit):
    mask = df[col].between(-limit, limit)
    sub = df.loc[mask, ["bin", col]].copy()
    if sub.empty:
        return np.full(len(bins), np.nan, dtype=float)

    sub["ok"] = sub[col].between(-window, window)
    grp = sub.groupby("bin")["ok"].agg(total="size", ok="sum")
    pct = (grp["ok"] / grp["total"]) * 100
    return pct.reindex(bins).to_numpy(dtype=float)


# ================================================================
# Bild laden (ACG Logo)
# ================================================================
def load_base64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None


# ================================================================
# MAIN APP
# ================================================================
def main():
    if not check_password():
        return

    # ------------------ Compact Toggle (ohne Sidebar) ------------------
    top_left, top_right = st.columns([1, 2])
    with top_left:
        compact = st.checkbox("Kompaktmodus (mobile)", False, key="compact_mode")

    # ------------------ Globales Styling ----------------
    base_css = """
    <style>
      .stApp { background-color: #f5f7fb; }
      .acg-panel {
        background: #fff; padding: 1.2rem 1.5rem; border-radius: 0.75rem;
        border: 1px solid #e0e0e0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1.2rem;
      }
      .acg-muted { color:#666; font-size:0.85rem; }

      .acg-header {
        display:flex; align-items:center; background:#003DA5;
        padding:20px 30px; border-radius:12px; margin-bottom:25px; color:white;
      }
      .acg-header .title { font-size:40px; font-weight:700; color:white; }
      .acg-header img.logo-desktop { height:120px; margin-right:30px; display:block; }
      .acg-header img.logo-mobile { height:72px; margin-bottom:10px; display:none; }

      .acg-footer {
        text-align:center; color:#666; font-size:0.85rem;
        padding:12px 0; margin-top:18px; border-top:1px solid #eee;
      }

      @media (max-width: 600px) {
        .acg-header { flex-direction: column; padding:12px; }
        .acg-header .title { font-size:20px; text-align:center; }
        .acg-header img.logo-desktop { display:none; }
        .acg-header img.logo-mobile { display:block; }
        .acg-panel { padding:0.8rem 0.9rem; margin-bottom:0.8rem; }
      }
    </style>
    """
    st.markdown(base_css, unsafe_allow_html=True)

    if compact:
        st.markdown(
            """
            <style>
              .acg-panel { padding: 0.6rem 0.8rem !important; margin-bottom:0.6rem !important; }
              .acg-header { padding:10px 12px !important; }
              .acg-header .title { font-size:18px !important; }
              .acg-header img.logo-desktop { height:70px !important; margin-right:12px !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )

    # ------------------ Header + Logo -------------------
    logo_b64 = load_base64("acg_logo.png")
    logo_small_b64 = load_base64("acg_logo_small.png")

    if logo_b64 or logo_small_b64:
        img_desktop = f'<img class="logo-desktop" src="data:image/png;base64,{logo_b64}" alt="logo">' if logo_b64 else ""
        img_mobile = f'<img class="logo-mobile" src="data:image/png;base64,{logo_small_b64 or logo_b64}" alt="logo">' if (logo_small_b64 or logo_b64) else ""
        st.markdown(
            f"""
            <div class="acg-header" role="banner">
              {img_desktop}
              {img_mobile}
              <div class="title">CDM Delta Analysis Dashboard</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="acg-header" role="banner">
              <div class="title">CDM Delta Analysis Dashboard</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ------------------ Daten laden ---------------------
    try:
        df = load_data()
        loaded_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    except Exception as e:
        st.error(f"Daten konnten nicht geladen werden: {e}")
        return

    # ------------------ Global Filter: Zeitbereich -----------------
    with st.container():
        st.markdown('<div class="acg-panel">', unsafe_allow_html=True)

        t_min, t_max = st.slider(
            "Zeitbereich vor ATOT (min) – min/max",
            min_value=0,
            max_value=240,
            value=(0, 120),
            step=5,
            key="time_range",
        )

        if t_min == t_max:
            st.caption(f"Auswahl: **genau Bin {t_min}**")
        else:
            st.caption(f"Auswahl: **{t_min} – {t_max}** Minuten vor ATOT")

        st.markdown("</div>", unsafe_allow_html=True)

    df = df[(df[COL_MIN_TO_ATOT] >= t_min) & (df[COL_MIN_TO_ATOT] <= t_max)].copy()

    # ------------------ Statistiken (ABS) ---------------------
    etot_stats = compute_stats(df, COL_ETOT, DELTA_LIMIT)
    ctot_stats = compute_stats(df, COL_CTOT, DELTA_LIMIT)
    atc_stats = compute_stats(df, COL_ATC, DELTA_LIMIT)

    etot_counts = etot_stats["count"]
    ctot_counts = ctot_stats["count"].reindex(etot_stats.index).fillna(0)
    atc_counts = atc_stats["count"].reindex(etot_stats.index).fillna(0)

    ratio_ctot = (ctot_counts / etot_counts.replace(0, np.nan)) * 100
    ratio_atc = (atc_counts / etot_counts.replace(0, np.nan)) * 100

    df_etot = df[df[COL_ETOT].notna() & df[COL_ETOT].between(-DELTA_LIMIT, DELTA_LIMIT)]

    def n0_from_stats(stats):
        if 0 in stats.index:
            return int(stats.loc[0, "count"])
        return 0

    n0_etot = n0_from_stats(etot_stats)
    n0_ctot = n0_from_stats(ctot_stats)
    n0_atc = n0_from_stats(atc_stats)

    cat_counts0 = df_etot[df_etot["bin"] == 0].groupby("AirlineCategory")[COL_ETOT].size()
    rw_counts0 = df_etot[df_etot["bin"] == 0].groupby("Runway")[COL_ETOT].size()

    fig_w, fig_h = (7, 3.5) if compact else (10, 5)

    # ================================================================
    # TABS (Variante A)
    # ================================================================
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Panel 1 – Mean",
        "Panel 2 – Stabilität",
        "Panel 3 – Airlines",
        "Panel 4 – Runways",
        "Panel 5 – Histogramm",
        "Export",
    ])

    # ================================================================
    # TAB 1 – Mean-Verläufe
    # ================================================================
    with tab1:
        st.markdown('<div class="acg-panel">', unsafe_allow_html=True)
        st.subheader("Panel 1 – Mean-Verläufe ETOT / CTOT / ATC-TTOT")

        c1, c2, c3 = st.columns(3)
        with c1:
            s_etot = st.checkbox("ETOT anzeigen", key="p1_etot")
        with c2:
            s_ctot = st.checkbox("CTOT anzeigen", key="p1_ctot")
        with c3:
            s_atc = st.checkbox("ATC-TTOT anzeigen", key="p1_atc")

        fig1, ax1 = plt.subplots(figsize=(fig_w, fig_h))

        if s_etot:
            valid = etot_stats["count"] >= MIN_COUNT
            ax1.plot(
                etot_stats.index[valid],
                smooth(etot_stats.loc[valid, "mean"]),
                marker="o",
                linewidth=2,
                color=colors["etot"],
                label="ETOT (Abs)",
            )

        if s_ctot:
            valid = ctot_stats["count"] >= MIN_COUNT
            ax1.plot(
                ctot_stats.index[valid],
                smooth(ctot_stats.loc[valid, "mean"]),
                marker="o",
                linewidth=2,
                color=colors["ctot"],
                label="CTOT (Abs)",
            )

        if s_atc:
            valid = atc_stats["count"] >= MIN_COUNT
            ax1.plot(
                atc_stats.index[valid],
                smooth(atc_stats.loc[valid, "mean"]),
                marker="o",
                linewidth=2,
                color=colors["atc"],
                label="ATC-TTOT (Abs)",
            )

        # Info-Box unten rechts
        etot_counts_box = etot_stats["count"]
        ctot_counts_box = ctot_stats["count"].reindex(etot_stats.index).fillna(0)
        atc_counts_box = atc_stats["count"].reindex(etot_stats.index).fillna(0)

        ratio_ctot_box = np.where(
            etot_counts_box > 0, (ctot_counts_box / etot_counts_box) * 100, np.nan
        )
        ratio_atc_box = np.where(
            etot_counts_box > 0, (atc_counts_box / etot_counts_box) * 100, np.nan
        )

        bins_arr = etot_stats.index.to_numpy()

        valid_ct = ~np.isnan(ratio_ctot_box)
        ct_start = float(ratio_ctot_box[valid_ct][0]) if valid_ct.any() else np.nan
        ct_end = float(ratio_ctot_box[valid_ct][-1]) if valid_ct.any() else np.nan

        valid_at = ~np.isnan(ratio_atc_box)
        at_start = float(ratio_atc_box[valid_at][0]) if valid_at.any() else np.nan

        thr_bin = None
        thr_mask = valid_at & (ratio_atc_box < 10)
        if thr_mask.any():
            thr_bin = int(bins_arr[thr_mask][0])

        lines = ["Datenbasis (Anteil Flüge)", "──────────────────────"]
        if not np.isnan(ct_start):
            lines += [f"CTOT vorhanden bei {ct_start:.0f}%", "der Flüge bei ATOT"]
        if not np.isnan(ct_end):
            lines += [f"→ ca. {ct_end:.0f}% der Flüge", f"im Bereich {int(t_min)}–{int(t_max)} min vor ATOT"]
        lines.append("")
        if not np.isnan(at_start):
            lines += [f"ATC-TTOT vorhanden bei {at_start:.0f}%", "der Flüge bei ATOT"]
        if thr_bin is not None:
            lines += [f"→ ab ca. {thr_bin} min vor ATOT", "keine verwertbaren ATC-TTOT mehr"]

        props = dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="black", alpha=0.85)
        ax1.text(
            0.98, 0.05, "\n".join(lines),
            transform=ax1.transAxes,
            fontsize=11,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=props,
        )

        ax1.grid(True)
        ax1.legend()
        ax1.set_xlabel("Min vor ATOT")
        ax1.set_ylabel("Delta (min)")
        fig1.tight_layout()

        st.pyplot(fig1)
        st.markdown("</div>", unsafe_allow_html=True)

    # ================================================================
    # TAB 2 – Stabilität
    # ================================================================
    with tab2:
        st.markdown('<div class="acg-panel">', unsafe_allow_html=True)
        st.subheader("Panel 2 – Stabilität (± Zeitfenster)")

        window = st.slider("Fenster (± Minuten)", 1, 15, 3, key="p2_window")
        bins = etot_stats.index

        pct_etot = percent_within_window(df, bins, COL_ETOT, window, DELTA_LIMIT)
        pct_ctot = percent_within_window(df, bins, COL_CTOT, window, DELTA_LIMIT)
        pct_atc = percent_within_window(df, bins, COL_ATC, window, DELTA_LIMIT)

        fig2, ax2 = plt.subplots(figsize=(fig_w, fig_h))
        ax2.plot(bins, pct_etot, marker="o", color=colors["etot"], label=f"ETOT ±{window} min (n={n0_etot})")
        ax2.plot(bins, pct_ctot, marker="o", color=colors["ctot"], label=f"CTOT ±{window} min (n={n0_ctot})")
        ax2.plot(bins, pct_atc, marker="o", color=colors["atc"], label=f"ATC-TTOT ±{window} min (n={n0_atc})")

        ax2.set_ylim(0, 100)
        ax2.grid(True)
        ax2.legend()
        ax2.set_xlabel("Min vor ATOT")
        ax2.set_ylabel("Anteil (%)")
        fig2.tight_layout()

        st.pyplot(fig2)
        st.markdown("</div>", unsafe_allow_html=True)

    # ================================================================
    # TAB 3 – Airline Kategorien
    # ================================================================
    with tab3:
        st.markdown('<div class="acg-panel">', unsafe_allow_html=True)
        st.subheader("Panel 3 – Airline-Kategorien (ETOT)")

        cols = st.columns(len(CATEGORIES_OF_INTEREST))
        show_cat = {}
        for i, cat in enumerate(CATEGORIES_OF_INTEREST):
            with cols[i]:
                show_cat[cat] = st.checkbox(cat, False, key=f"p3_cat_{cat}")

        fig3, ax3 = plt.subplots(figsize=(fig_w, fig_h))
        cat_grp = df_etot.groupby(["bin", "AirlineCategory"])[COL_ETOT].mean()

        for cat in CATEGORIES_OF_INTEREST:
            if not show_cat.get(cat, False):
                continue
            if cat not in cat_grp.index.get_level_values(1):
                continue
            series = cat_grp.xs(cat, level="AirlineCategory").sort_index()
            n0_cat = int(cat_counts0.get(cat, 0))
            ax3.plot(series.index, smooth(series), marker="o", linewidth=2, label=f"{cat} (n={n0_cat})")

        ax3.grid(True)
        ax3.legend()
        ax3.set_xlabel("Min vor ATOT")
        ax3.set_ylabel("Delta ETOT (min)")
        fig3.tight_layout()

        st.pyplot(fig3)
        st.markdown("</div>", unsafe_allow_html=True)

    # ================================================================
    # TAB 4 – Runways
    # ================================================================
    with tab4:
        st.markdown('<div class="acg-panel">', unsafe_allow_html=True)
        st.subheader("Panel 4 – Runways (ETOT)")

        cols = st.columns(len(RUNWAYS_OF_INTEREST))
        show_rw = {}
        for i, rw in enumerate(RUNWAYS_OF_INTEREST):
            with cols[i]:
                show_rw[rw] = st.checkbox(f"RWY {rw}", False, key=f"p4_rw_{rw}")

        fig4, ax4 = plt.subplots(figsize=(fig_w, fig_h))
        rw_grp = df_etot.groupby(["bin", "Runway"])[COL_ETOT].mean()

        for rw in RUNWAYS_OF_INTEREST:
            if not show_rw.get(rw, False):
                continue
            if rw not in rw_grp.index.get_level_values(1):
                continue
            series = rw_grp.xs(rw, level="Runway")
            n0_rw = int(rw_counts0.get(rw, 0))
            ax4.plot(series.index, smooth(series), marker="o", linewidth=2, label=f"RWY {rw} (n={n0_rw})")

        ax4.grid(True)
        ax4.legend()
        ax4.set_xlabel("Min vor ATOT")
        ax4.set_ylabel("Delta ETOT (min)")
        fig4.tight_layout()

        st.pyplot(fig4)
        st.markdown("</div>", unsafe_allow_html=True)

    # ================================================================
    # TAB 5 – Histogramm Signed
    # ================================================================
    with tab5:
        st.markdown('<div class="acg-panel">', unsafe_allow_html=True)
        st.subheader("Panel 5 – Histogramm (DeltaSigned)")

        c1, c2, c3 = st.columns(3)
        with c1:
            h_etot = st.checkbox("ETOT (Signed) anzeigen", True, key="p5_etot")
        with c2:
            h_ctot = st.checkbox("CTOT (Signed) anzeigen", True, key="p5_ctot")
        with c3:
            h_atc = st.checkbox("ATC-TTOT (Signed) anzeigen", False, key="p5_atc")

        # NEW: Mean-Bias Linien Toggle
        show_mean_bias = st.checkbox("Mean Bias anzeigen (vertikale Linie)", True, key="p5_meanbias")

        hist_bin_width = st.slider("Histogramm-Breite (min)", 1, 10, 2, step=1, key="p5_bw")
        hist_limit = st.slider("Anzeige-Limit (± min)", 30, 180, 120, step=10, key="p5_lim")

        bins_hist = np.arange(-hist_limit, hist_limit + hist_bin_width, hist_bin_width)

        fig5, ax5 = plt.subplots(figsize=(fig_w, fig_h))

        def plot_hist_with_mean(col, label, color, mean_linestyle="--"):
            s = df[col].dropna()
            s = s[(s >= -hist_limit) & (s <= hist_limit)]

            # --- Perzentil-Cut gegen Ausreißer ---
            low, high = np.percentile(s, [5, 95])
            s = s[(s >= low) & (s <= high)]

            if len(s) == 0:
                return

            mu = float(s.mean())

            ax5.hist(
                s,
                bins=bins_hist,
                alpha=0.45,
                density=True,
                label=f"{label} (n={len(s)}, μ={mu:.2f})" if show_mean_bias else f"{label} (n={len(s)})",
                color=color,
                edgecolor="none",
            )

            if show_mean_bias:
                ax5.axvline(mu, linestyle=mean_linestyle, linewidth=2, color=color)

        if h_etot:
            plot_hist_with_mean(COL_ETOT_S, "ETOT", colors["etot"])
        if h_ctot:
            plot_hist_with_mean(COL_CTOT_S, "CTOT", colors["ctot"])
        if h_atc:
            plot_hist_with_mean(COL_ATC_S, "ATC-TTOT", colors["atc"])

        ax5.axvline(0, linewidth=1)
        ax5.grid(True)
        ax5.set_xlabel("DeltaSigned (min)")
        ax5.set_ylabel("Dichte")
        ax5.legend()
        fig5.tight_layout()

        st.pyplot(fig5)
        st.markdown("</div>", unsafe_allow_html=True)

    # ================================================================
    # TAB 6 – Export
    # ================================================================
    with tab6:
        st.markdown('<div class="acg-panel">', unsafe_allow_html=True)
        st.subheader("Excel Export – Summary")

        summary = pd.DataFrame({
            "bin": etot_stats.index,
            "ETOT_mean": etot_stats["mean"],
            "ETOT_count": etot_stats["count"],
            "CTOT_mean": ctot_stats["mean"].reindex(etot_stats.index),
            "CTOT_count": ctot_stats["count"].reindex(etot_stats.index),
            "ATC_mean": atc_stats["mean"].reindex(etot_stats.index),
            "ATC_count": atc_stats["count"].reindex(etot_stats.index),
            "CTOT_ETOT_ratio_%": ratio_ctot,
            "ATC_ETOT_ratio_%": ratio_atc,
        })

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as w:
            summary.to_excel(w, index=False, sheet_name="Summary")

        st.download_button(
            "Excel-Summary herunterladen",
            data=output.getvalue(),
            file_name=f"summary_{t_min}-{t_max}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # ================================================================
    # Footer
    # ================================================================
    st.markdown(
        f"""
        <div class="acg-footer" role="contentinfo">
          <div>© DZ · Stand: {loaded_at}</div>
          <div class="acg-muted">Datenquelle: B2B CDM Daten von 10.-23.11.2025</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
