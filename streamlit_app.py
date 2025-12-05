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
            <h1 style="color:#003DA5; margin-bottom:0.2rem;">
                CDM Dashboard – Login
            </h1>
            <p style="color:#555; font-size:0.95rem;">
                Bitte Passwort eingeben, um das Dashboard zu öffnen.
            </p>
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
    "DLH Group": ["AUA","SWR","EWG","DLH","BEL","CLH","DLA","OCN"],
    "Low Cost Carrier": ["RYR","WZZ","WMT","EZY","EZS","EJU","VLG","EXS","TRA","TVF","CAI","CXI","SXS","PGT","TKJ"],
    "Long Haul": ["UAE","QTR","ETD","ACA","EVA","ETH","CHH","CAL","KAL","JAL","AIC","ABY","CCA"],
    "Biz Jets": ["VJT","NJE","AWH","GDK","AOJ","LDX","VCJ","JFL","TJS","PTN","GCK","IFA","IJM","UAG","FSF","PAV","PVD","BTX","TOY","MPC","OEE","OEH","OEF","OEI"],
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
# Daten laden (GitHub raw URL kommt aus secrets)
# ================================================================
@st.cache_data
def load_data():
    """Lädt die Excel-Datei von der in st.secrets konfigurierten URL.
    Bei Fehlern wird eine RuntimeError ausgelöst."""
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

    numeric_cols = [
        "Min bis ATOT",
        "Delta - ETOT (min)",
        "Delta - CTOT (min)",
        "Delta - ATC TTOT (min)"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Min bis ATOT"])

    df["bin"] = (df["Min bis ATOT"] / BIN_SIZE).astype(int) * BIN_SIZE
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
    """Vectorisierte Berechnung: für jeden bin den Anteil (%) von Werten in ±window
    (nur Werte innerhalb ±limit werden berücksichtigt)."""
    mask = df[col].between(-limit, limit)
    sub = df.loc[mask, ["bin", col]].copy()
    if sub.empty:
        return np.full(len(bins), np.nan, dtype=float)
    sub["ok"] = sub[col].between(-window, window)
    grp = sub.groupby("bin")["ok"].agg(total="size", ok="sum")
    pct = (grp["ok"] / grp["total"]) * 100
    # Reindex auf bins, fehlende Bins -> NaN
    pct_full = pct.reindex(bins).to_numpy(dtype=float)
    return pct_full


# ================================================================
# Bild laden (ACG Logo)
# ================================================================
def load_base64(path):
    """Lädt eine lokale Bilddatei als base64-String; gibt None bei Fehlern zurück."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None


# ================================================================
# MAIN APP
# ================================================================
def main():

    # ------------------ Passwortschutz ------------------
    if not check_password():
        return

    # ------------------ Compact-Modus (Sidebar) ------------------
    compact = st.sidebar.checkbox("Kompaktmodus (mobile)", False)

    # ------------------ Globales Styling ----------------
    base_css = """
        <style>
            .stApp { background-color: #f5f7fb; }
            .acg-panel {
                background: #fff;
                padding: 1.2rem 1.5rem;
                border-radius: 0.75rem;
                border: 1px solid #e0e0e0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                margin-bottom: 1.2rem;
            }
            .acg-muted { color:#666; font-size:0.85rem; }

            /* Header */
            .acg-header {
                display:flex;
                align-items:center;
                background:#003DA5;
                padding:20px 30px;
                border-radius:12px;
                margin-bottom:35px;
                color:white;
            }
            .acg-header .title {
                font-size:40px;
                font-weight:700;
                color:white;
            }
            .acg-header img.logo-desktop {
                height:120px;
                margin-right:30px;
                display:block;
            }
            .acg-header img.logo-mobile {
                height:72px;
                margin-bottom:10px;
                display:none;
            }

            /* Footer */
            .acg-footer {
                text-align:center;
                color:#666;
                font-size:0.85rem;
                padding:12px 0;
                margin-top:18px;
                border-top:1px solid #eee;
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
        st.markdown("""
            <style>
                .acg-panel { padding: 0.6rem 0.8rem !important; margin-bottom:0.6rem !important; }
                .acg-header { padding:10px 12px !important; }
                .acg-header .title { font-size:18px !important; }
                .acg-header img.logo-desktop { height:70px !important; margin-right:12px !important; }
            </style>
        """, unsafe_allow_html=True)

    # ------------------ Header + Logo -------------------
    logo_b64 = load_base64("acg_logo.png")
    logo_small_b64 = load_base64("acg_logo_small.png")

    if logo_b64 or logo_small_b64:
        img_desktop = f'<img class="logo-desktop" src="data:image/png;base64,{logo_b64}" alt="logo">' if logo_b64 else ""
        img_mobile = f'<img class="logo-mobile" src="data:image/png;base64,{logo_small_b64 or logo_b64}" alt="logo">' if (logo_small_b64 or logo_b64) else ""
        st.markdown(f"""
            <div class="acg-header" role="banner">
                {img_desktop}
                {img_mobile}
                <div class="title">CDM Delta Analysis Dashboard</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="acg-header" role="banner">
                <div class="title">CDM Delta Analysis Dashboard</div>
            </div>
        """, unsafe_allow_html=True)

    # ------------------ Daten laden ---------------------
    try:
        df = load_data()
        loaded_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    except Exception as e:
        st.error(f"Daten konnten nicht geladen werden: {e}")
        return

    # ------------------ TIME_MAX Slider -----------------
    with st.container():
        st.markdown('<div class="acg-panel">', unsafe_allow_html=True)
        time_max = st.slider(
            "Maximale Zeit vor ATOT (min)",
            60, 240, 120, step=5
        )
        st.markdown("</div>", unsafe_allow_html=True)

    df = df[(df["Min bis ATOT"] >= TIME_MIN) & (df["Min bis ATOT"] <= time_max)].copy()

    # ------------------ Statistiken ---------------------
    etot_stats = compute_stats(df, "Delta - ETOT (min)", DELTA_LIMIT)
    ctot_stats = compute_stats(df, "Delta - CTOT (min)", DELTA_LIMIT)
    atc_stats  = compute_stats(df, "Delta - ATC TTOT (min)", DELTA_LIMIT)

    etot_counts = etot_stats["count"]
    ctot_counts = ctot_stats["count"].reindex(etot_stats.index).fillna(0)
    atc_counts  = atc_stats["count"].reindex(etot_stats.index).fillna(0)

    # --- Gesamt-n je Serie (für Textfelder / Panels) ---
    total_etot = int(etot_counts.sum())
    total_ctot = int(ctot_counts.sum())
    total_atc  = int(atc_counts.sum())

    ratio_ctot = (ctot_counts / etot_counts.replace(0, np.nan)) * 100
    ratio_atc = (atc_counts / etot_counts.replace(0, np.nan)) * 100

    df_etot = df[
        df["Delta - ETOT (min)"].notna() &
        df["Delta - ETOT (min)"].between(-DELTA_LIMIT, DELTA_LIMIT)
    ]

    # figure sizes based on compact mode
    if compact:
        fig_w, fig_h = 7, 3.5
    else:
        fig_w, fig_h = 10, 5

    # ================================================================
    # PANEL 1 – Mean Verläufe
    # ================================================================
    st.markdown('<div class="acg-panel">', unsafe_allow_html=True)
    st.subheader("Panel 1 – Mean-Verläufe ETOT / CTOT / ATC-TTOT")
    st.markdown(
        f'<p class="acg-muted">Datenbasis: ETOT n={total_etot}, CTOT n={total_ctot}, ATC-TTOT n={total_atc}</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        s_etot = st.checkbox("ETOT anzeigen")
    with col2:
        s_ctot = st.checkbox("CTOT anzeigen")
    with col3:
        s_atc = st.checkbox("ATC-TTOT anzeigen")

    fig1, ax1 = plt.subplots(figsize=(fig_w, fig_h))

    if s_etot:
        valid = etot_stats["count"] >= MIN_COUNT
        ax1.plot(
            etot_stats.index[valid],
            smooth(etot_stats.loc[valid, "mean"]),
            marker="o", linewidth=2, color=colors["etot"], label=f"ETOT (n={total_etot})"
        )

    if s_ctot:
        valid = ctot_stats["count"] >= MIN_COUNT
        ax1.plot(
            ctot_stats.index[valid],
            smooth(ctot_stats.loc[valid, "mean"]),
            marker="o", linewidth=2, color=colors["ctot"], label=f"CTOT (n={total_ctot})"
        )

    if s_atc:
        valid = atc_stats["count"] >= MIN_COUNT
        ax1.plot(
            atc_stats.index[valid],
            smooth(atc_stats.loc[valid, "mean"]),
            marker="o", linewidth=2, color=colors["atc"], label=f"ATC-TTOT (n={total_atc})"
        )

    ax1.grid(True)
    ax1.legend()
    ax1.set_xlabel("Min vor ATOT")
    ax1.set_ylabel("Delta (min)")
    fig1.tight_layout()
    st.pyplot(fig1)
    st.markdown("</div>", unsafe_allow_html=True)

    # ================================================================
    # PANEL 2 – Stabilität ± Window
    # ================================================================
    st.markdown('<div class="acg-panel">', unsafe_allow_html=True)
    st.subheader("Panel 2 – Stabilität (± Zeitfenster)")
    st.markdown(
        f'<p class="acg-muted">Datenbasis: ETOT n={total_etot}, CTOT n={total_ctot}, ATC-TTOT n={total_atc}</p>',
        unsafe_allow_html=True,
    )

    window = st.slider("Fenster (± Minuten)", 1, 15, 3)

    bins = etot_stats.index
    pct_etot = percent_within_window(df, bins, "Delta - ETOT (min)", window, DELTA_LIMIT)
    pct_ctot = percent_within_window(df, bins, "Delta - CTOT (min)", window, DELTA_LIMIT)
    pct_atc = percent_within_window(df, bins, "Delta - ATC TTOT (min)", window, DELTA_LIMIT)

    fig2, ax2 = plt.subplots(figsize=(fig_w, fig_h))

    ax2.plot(bins, pct_etot, marker="o", color=colors["etot"], label=f"ETOT (n={total_etot})")
    ax2.plot(bins, pct_ctot, marker="o", color=colors["ctot"], label=f"CTOT (n={total_ctot})")
    ax2.plot(bins, pct_atc, marker="o", color=colors["atc"], label=f"ATC-TTOT (n={total_atc})")

    ax2.set_ylim(0, 100)
    ax2.grid(True)
    ax2.legend()
    ax2.set_xlabel("Min vor ATOT")
    ax2.set_ylabel("Anteil (%)")
    fig2.tight_layout()
    st.pyplot(fig2)
    st.markdown("</div>", unsafe_allow_html=True)

    # ================================================================
    # PANEL 3 – Airline Kategorien
    # ================================================================
    st.markdown('<div class="acg-panel">', unsafe_allow_html=True)
    st.subheader("Panel 3 – Airline-Kategorien (ETOT)")

    cols = st.columns(len(CATEGORIES_OF_INTEREST))
    show_cat = {}
    for i, cat in enumerate(CATEGORIES_OF_INTEREST):
        with cols[i]:
            show_cat[cat] = st.checkbox(cat, False)

    fig3, ax3 = plt.subplots(figsize=(fig_w, fig_h))
    cat_grp = df_etot.groupby(["bin", "AirlineCategory"])["Delta - ETOT (min)"].mean()

    # n je Airline-Kategorie (gesamt im gefilterten Datensatz)
    cat_counts = df_etot.groupby("AirlineCategory")["Delta - ETOT (min)"].count()

    for cat in CATEGORIES_OF_INTEREST:
        if not show_cat[cat]:
            continue
        if cat not in cat_grp.index.get_level_values(1):
            continue

        series = cat_grp.xs(cat, level="AirlineCategory").sort_index()
        n_cat = int(cat_counts.get(cat, 0))
        ax3.plot(
            series.index,
            smooth(series),
            marker="o",
            linewidth=2,
            label=f"{cat} (n={n_cat})"
        )

    ax3.grid(True)
    ax3.legend()
    ax3.set_xlabel("Min vor ATOT")
    ax3.set_ylabel("Delta ETOT (min)")
    fig3.tight_layout()
    st.pyplot(fig3)
    st.markdown("</div>", unsafe_allow_html=True)

    # ================================================================
    # PANEL 4 – Runways
    # ================================================================
    st.markdown('<div class="acg-panel">', unsafe_allow_html=True)
    st.subheader("Panel 4 – Runways (ETOT)")

    cols = st.columns(len(RUNWAYS_OF_INTEREST))
    show_rw = {}
    for i, rw in enumerate(RUNWAYS_OF_INTEREST):
        with cols[i]:
            show_rw[rw] = st.checkbox(f"RWY {rw}", False)

    fig4, ax4 = plt.subplots(figsize=(fig_w, fig_h))
    rw_grp = df_etot.groupby(["bin", "Runway"])["Delta - ETOT (min)"].mean()

    # n je Runway (gesamt im gefilterten Datensatz)
    rw_counts = df_etot.groupby("Runway")["Delta - ETOT (min)"].count()

    for rw in RUNWAYS_OF_INTEREST:
        if not show_rw[rw]:
            continue
        if rw not in rw_grp.index.get_level_values(1):
            continue
        series = rw_grp.xs(rw, level="Runway").sort_index()
        n_rw = int(rw_counts.get(rw, 0))
        ax4.plot(
            series.index,
            smooth(series),
            marker="o",
            linewidth=2,
            label=f"RWY {rw} (n={n_rw})"
        )

    ax4.grid(True)
    ax4.legend()
    ax4.set_xlabel("Min vor ATOT")
    ax4.set_ylabel("Delta ETOT (min)")
    fig4.tight_layout()
    st.pyplot(fig4)
    st.markdown("</div>", unsafe_allow_html=True)

    # ================================================================
    # Export – Summary
    # ================================================================
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
        file_name="summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ================================================================
    # Footer
    # ================================================================
    st.markdown(f"""
        <div class="acg-footer" role="contentinfo">
            <div>© Sascha Derp · Stand: {loaded_at}</div>
            <div class="acg-muted"> Datenquelle: B2B CDM Daten von 10.-23.11.2025 </div>
        </div>
    """, unsafe_allow_html=True)


# ================================================================
# Start App
# ================================================================
if __name__ == "__main__":
    main()
