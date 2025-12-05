import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
import streamlit as st
import io
import requests
import base64

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
    url = st.secrets["file_links"]["xlsm_url"]
    resp = requests.get(url)
    resp.raise_for_status()

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
    result = []
    for b in bins:
        sub = df[df["bin"] == b]
        sub = sub[sub[col].between(-limit, limit)]
        if len(sub) == 0:
            result.append(np.nan)
            continue
        ok = sub[sub[col].between(-window, window)]
        result.append((len(ok) / len(sub)) * 100)
    return np.array(result)


# ================================================================
# Bild laden (ACG Logo)
# ================================================================
def load_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ================================================================
# MAIN APP
# ================================================================
def main():

    # ------------------ Passwortschutz ------------------
    if not check_password():
        return

    # ------------------ Globales Styling ----------------
    st.markdown("""
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
        </style>
    """, unsafe_allow_html=True)

    # ------------------ Header + Logo -------------------
    logo_b64 = load_base64("acg_logo.png")

    st.markdown(f"""
        <div style="display:flex; align-items:center;
                    background:#003DA5; padding:20px 30px;
                    border-radius:12px; margin-bottom:35px;">
            <img src="data:image/png;base64,{logo_b64}"
                 style="height:120px; margin-right:30px;">
            <div style="font-size:40px; font-weight:700; color:white;">
                CDM Delta Analysis Dashboard
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ------------------ Daten laden ---------------------
    df = load_data()

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

    ratio_ctot = (ctot_counts / etot_counts.replace(0, np.nan)) * 100
    ratio_atc = (atc_counts / etot_counts.replace(0, np.nan)) * 100

    df_etot = df[
        df["Delta - ETOT (min)"].notna() &
        df["Delta - ETOT (min)"].between(-DELTA_LIMIT, DELTA_LIMIT)
    ]

    # ================================================================
    # PANEL 1 – Mean Verläufe
    # ================================================================
    st.markdown('<div class="acg-panel">', unsafe_allow_html=True)
    st.subheader("Panel 1 – Mean-Verläufe ETOT / CTOT / ATC-TTOT")

    col1, col2, col3 = st.columns(3)
    with col1:
        s_etot = st.checkbox("ETOT anzeigen")
    with col2:
        s_ctot = st.checkbox("CTOT anzeigen")
    with col3:
        s_atc = st.checkbox("ATC-TTOT anzeigen")

    fig1, ax1 = plt.subplots(figsize=(10, 5))

    if s_etot:
        valid = etot_stats["count"] >= MIN_COUNT
        ax1.plot(
            etot_stats.index[valid],
            smooth(etot_stats.loc[valid, "mean"]),
            marker="o", linewidth=2, color=colors["etot"], label="ETOT"
        )

    if s_ctot:
        valid = ctot_stats["count"] >= MIN_COUNT
        ax1.plot(
            ctot_stats.index[valid],
            smooth(ctot_stats.loc[valid, "mean"]),
            marker="o", linewidth=2, color=colors["ctot"], label="CTOT"
        )

    if s_atc:
        valid = atc_stats["count"] >= MIN_COUNT
        ax1.plot(
            atc_stats.index[valid],
            smooth(atc_stats.loc[valid, "mean"]),
            marker="o", linewidth=2, color=colors["atc"], label="ATC-TTOT"
        )

    ax1.grid(True)
    ax1.legend()
    ax1.set_xlabel("Min vor ATOT")
    ax1.set_ylabel("Delta (min)")

    st.pyplot(fig1)
    st.markdown("</div>", unsafe_allow_html=True)

    # ================================================================
    # PANEL 2 – Stabilität ± Window
    # ================================================================
    st.markdown('<div class="acg-panel">', unsafe_allow_html=True)
    st.subheader("Panel 2 – Stabilität (± Zeitfenster)")

    window = st.slider("Fenster (± Minuten)", 1, 15, 3)

    bins = etot_stats.index
    pct_etot = percent_within_window(df, bins, "Delta - ETOT (min)", window, DELTA_LIMIT)
    pct_ctot = percent_within_window(df, bins, "Delta - CTOT (min)", window, DELTA_LIMIT)
    pct_atc = percent_within_window(df, bins, "Delta - ATC TTOT (min)", window, DELTA_LIMIT)

    fig2, ax2 = plt.subplots(figsize=(10, 5))

    ax2.plot(bins, pct_etot, marker="o", color=colors["etot"], label="ETOT")
    ax2.plot(bins, pct_ctot, marker="o", color=colors["ctot"], label="CTOT")
    ax2.plot(bins, pct_atc, marker="o", color=colors["atc"], label="ATC-TTOT")

    ax2.set_ylim(0, 100)
    ax2.grid(True)
    ax2.legend()
    ax2.set_xlabel("Min vor ATOT")
    ax2.set_ylabel("Anteil (%)")

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

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    cat_grp = df_etot.groupby(["bin", "AirlineCategory"])["Delta - ETOT (min)"].mean()

    for cat in CATEGORIES_OF_INTEREST:
        if not show_cat[cat]:
            continue
        if cat not in cat_grp.index.get_level_values(1):
            continue

        series = cat_grp.xs(cat, level="AirlineCategory").sort_index()
        ax3.plot(
            series.index,
            smooth(series),
            marker="o",
            linewidth=2,
            label=cat
        )

    ax3.grid(True)
    ax3.legend()
    ax3.set_xlabel("Min vor ATOT")
    ax3.set_ylabel("Delta ETOT (min)")

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

    fig4, ax4 = plt.subplots(figsize=(10, 5))
    rw_grp = df_etot.groupby(["bin", "Runway"])["Delta - ETOT (min)"].mean()

    for rw in RUNWAYS_OF_INTEREST:
        if not show_rw[rw]:
            continue
        if rw not in rw_grp.index.get_level_values(1):
            continue
        series = rw_grp.xs(rw, level="Runway")
        ax4.plot(series.index, smooth(series), marker="o", linewidth=2, label=f"RWY {rw}")

    ax4.grid(True)
    ax4.legend()
    ax4.set_xlabel("Min vor ATOT")
    ax4.set_ylabel("Delta ETOT (min)")

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
# Start App
# ================================================================
if __name__ == "__main__":
    main()
