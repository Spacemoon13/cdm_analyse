import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
import streamlit as st
import io
import requests

# ================================================================
# 0) PASSWORTSCHUTZ
# ================================================================

def check_password():
    """Einfache Passwortabfrage mit Streamlit-Secrets + Session-State."""

    def password_entered():
        """Wird aufgerufen, wenn der User das Passwort submitted."""
        pwd = st.session_state.get("password_input", "")
        correct_pwd = st.secrets["auth"]["password"]
        if pwd == correct_pwd:
            st.session_state["password_correct"] = True
            # Passwort-Eingabefeld wieder entfernen
            del st.session_state["password_input"]
        else:
            st.session_state["password_correct"] = False

    # Wenn bereits korrekt eingeloggt → direkt durchlassen
    if st.session_state.get("password_correct", False):
        return True

    # Login-UI anzeigen
    st.title("CDM Dashboard – Login")
    st.write("Bitte Passwort eingeben, um das Dashboard zu öffnen.")

    st.text_input(
        "Passwort",
        type="password",
        key="password_input",
        on_change=password_entered,
    )

    # Fehlermeldung, falls Passwort falsch war
    if st.session_state.get("password_correct") is False:
        st.error("Falsches Passwort.")

    return False


# ================================================================
# 1) FESTE BASIS-EINSTELLUNGEN
# ================================================================

SHEET_NAME = "Deltas"

BIN_SIZE = 5
TIME_MIN = 0          # bleibt fix, TIME_MAX kommt als Slider
DELTA_LIMIT = 120     # ETOT/CTOT/ATC werden auf ±DELTA_LIMIT min begrenzt
MIN_COUNT = 200       # Mindestanzahl Werte pro Bin zum Anzeigen

# Airline-Kategorien: IATA/ICAO-Codes zu Gruppen zuordnen
AIRLINE_CATEGORIES = {
    "DLH Group":        ["AUA", "SWR", "EWG", "DLH", "BEL", "CLH", "DLA", "OCN"],
    "Low Cost Carrier": ["RYR","WZZ","WMT","EZY","EZS","EJU","VLG","EXS","TRA",
                         "TVF","CAI","CXI","SXS","PGT","TKJ"],
    "Long Haul":        ["UAE","QTR","ETD","ACA","EVA","ETH","CHH","CAL","KAL",
                         "JAL","AIC","ABY","CCA"],
    "Biz Jets":         ["VJT","NJE","AWH","GDK","AOJ","LDX","VCJ","JFL","TJS",
                         "VJT","PTN","GCK","IFA","IJM","UAG","FSF","PAV","PVD",
                         "BTX","TOY","MPC","OEE","OEH","OEF","OEI"],
}

# Reihenfolge / Auswahl der Kategorien im Panel 3
CATEGORIES_OF_INTEREST = ["DLH Group", "Low Cost Carrier", "Long Haul", "Biz Jets"]

# Runways, die in Panel 4 angezeigt werden sollen
RUNWAYS_OF_INTEREST = ["11", "16", "29", "34"]

sns.set(style="whitegrid")

# ACG CI Farben
colors = {
    "etot": "#003DA5",   # ACG Blau
    "ctot": "#FF6900",   # ACG Orange
    "atc":  "#6E6E6E",   # ACG Grau
}

# einfache Glättungsfunktion (rolling mean)
def smooth(series, window=3):
    return series.rolling(window=window, center=True, min_periods=1).mean()


# ================================================================
# 2) DATEN LADEN (einmalig, gecached) – per direkter URL (z.B. GitHub raw)
# ================================================================

@st.cache_data
def load_data():
    # Direkte Datei-URL aus Secrets, z.B. GitHub raw
    url = st.secrets["file_links"]["xlsm_url"]

    # Datei holen
    resp = requests.get(url)
    resp.raise_for_status()  # Fehler, falls Download schiefgeht

    # Bytes in Datei-ähnlichen Puffer packen
    file_like = io.BytesIO(resp.content)

    # Jetzt ganz normal als Excel lesen
    df = pd.read_excel(
        file_like,
        sheet_name=SHEET_NAME,
        usecols=[
            "Min bis ATOT",
            "Delta - ETOT (min)",
            "Delta - CTOT (min)",
            "Delta - ATC TTOT (min)",
            "Airline",
            "Runway",
        ],
        engine="openpyxl",  # wichtig bei .xlsm
    )

    # numeric cast
    for col in [
        "Min bis ATOT",
        "Delta - ETOT (min)",
        "Delta - CTOT (min)",
        "Delta - ATC TTOT (min)"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Min bis ATOT"])

    # Bins erzeugen (noch ohne TIME_MAX-Zuschnitt)
    df["bin"] = (df["Min bis ATOT"] / BIN_SIZE).astype(int) * BIN_SIZE

    # Strings säubern
    df["Runway"] = df["Runway"].astype(str).str.strip()
    df["Airline"] = df["Airline"].astype(str).str.strip()

    # Airline → Kategorie mappen
    airline_to_cat = {}
    for cat, codes in AIRLINE_CATEGORIES.items():
        for code in codes:
            airline_to_cat[code] = cat

    df["AirlineCategory"] = df["Airline"].map(airline_to_cat).fillna("Other")

    return df


def compute_stats(data, col, delta_limit):
    """Mean & Count pro Bin, gefiltert auf ±delta_limit."""
    mask = data[col].notna() & data[col].between(-delta_limit, delta_limit)
    sub = data[mask]
    stats = sub.groupby("bin")[col].agg(mean="mean", count="count").sort_index()
    return stats


def percent_within_window(df, bins, delta_col, window, delta_limit):
    """Prozent Flüge je Bin, deren Delta innerhalb ±window min liegt."""
    result = []
    for b in bins:
        sub = df[df["bin"] == b]
        sub = sub[sub[delta_col].between(-delta_limit, delta_limit)]

        if len(sub) == 0:
            result.append(np.nan)
            continue

        count_ok = sub[sub[delta_col].between(-window, window)].shape[0]
        pct = (count_ok / len(sub)) * 100
        result.append(pct)
    return np.array(result)


# ================================================================
# 3) STREAMLIT APP
# ================================================================

def main():
    # -------- Passwortschutz als erstes --------
    if not check_password():
        return

    st.title("CDM Delta Analysis – ATOT Delta ETOT / CTOT / ATC TTOT")

    # ------------------------------------
    # Slider ganz oben: TIME_MAX
    # ------------------------------------
    time_max = st.slider(
        "Maximale Zeit vor ATOT in Minuten",
        min_value=60,
        max_value=240,
        value=120,
        step=5
    )

    df = load_data()

    # Zeitbereich dynamisch zuschneiden
    df = df[(df["Min bis ATOT"] >= TIME_MIN) & (df["Min bis ATOT"] <= time_max)].copy()

    # Statistiken (mit DELTA_LIMIT)
    etot_stats = compute_stats(df, "Delta - ETOT (min)", DELTA_LIMIT)
    ctot_stats = compute_stats(df, "Delta - CTOT (min)", DELTA_LIMIT)
    atc_stats  = compute_stats(df, "Delta - ATC TTOT (min)", DELTA_LIMIT)

    # Datenbasis für Ratios / Info-Box
    etot_counts = etot_stats["count"]
    ctot_counts = ctot_stats["count"].reindex(etot_stats.index).fillna(0)
    atc_counts  = atc_stats["count"].reindex(etot_stats.index).fillna(0)

    ratio_ctot = np.where(etot_counts > 0, (ctot_counts / etot_counts) * 100, np.nan)
    ratio_atc  = np.where(etot_counts > 0, (atc_counts  / etot_counts) * 100, np.nan)

    # Gefilterte ETOT-Daten für Panel 3/4
    df_etot = df[
        df["Delta - ETOT (min)"].notna() &
        df["Delta - ETOT (min)"].between(-DELTA_LIMIT, DELTA_LIMIT)
    ].copy()

    # ================================================================
    # PANEL 1 — Mean-Verläufe
    # ================================================================
    st.subheader("Panel 1 – Mean-Verläufe ETOT / CTOT / ATC-TTOT")

    col_p1_1, col_p1_2, col_p1_3 = st.columns(3)
    with col_p1_1:
        show_p1_etot = st.checkbox("ETOT in Panel 1", value=False)
    with col_p1_2:
        show_p1_ctot = st.checkbox("CTOT in Panel 1", value=False)
    with col_p1_3:
        show_p1_atc = st.checkbox("ATC-TTOT in Panel 1", value=False)

    fig1, ax1 = plt.subplots(figsize=(10, 5))

    etot_valid = etot_stats["count"] >= MIN_COUNT
    ctot_valid = ctot_stats["count"] >= MIN_COUNT
    atc_valid  = atc_stats["count"]  >= MIN_COUNT

    # ETOT Mean (smoothed)
    if show_p1_etot and etot_valid.any():
        x_et = etot_stats.index[etot_valid]
        y_et = smooth(etot_stats.loc[etot_valid, "mean"])
        ax1.plot(
            x_et,
            y_et,
            marker="o", linewidth=2, color=colors["etot"],
            label="Mean ETOT"
        )

    # CTOT Mean (smoothed)
    if show_p1_ctot and ctot_valid.any():
        x_ct = ctot_stats.index[ctot_valid]
        y_ct = smooth(ctot_stats.loc[ctot_valid, "mean"])
        ax1.plot(
            x_ct,
            y_ct,
            marker="o", linewidth=2, color=colors["ctot"],
            label="Mean CTOT"
        )

    # ATC TTOT Mean (smoothed)
    if show_p1_atc and atc_valid.any():
        x_at = atc_stats.index[atc_valid]
        y_at = smooth(atc_stats.loc[atc_valid, "mean"])
        ax1.plot(
            x_at,
            y_at,
            marker="o", linewidth=2, color=colors["atc"],
            label="Mean ATC TTOT"
        )

    # Info-Box unten rechts
    etot_counts_box = etot_stats["count"]
    ctot_counts_box = ctot_stats["count"].reindex(etot_stats.index).fillna(0)
    atc_counts_box  = atc_stats["count"].reindex(etot_stats.index).fillna(0)

    ratio_ctot_box = np.where(
        etot_counts_box > 0, (ctot_counts_box / etot_counts_box) * 100, np.nan
    )
    ratio_atc_box = np.where(
        etot_counts_box > 0, (atc_counts_box / etot_counts_box) * 100, np.nan
    )

    bins_arr = etot_stats.index.to_numpy()

    valid_ct = ~np.isnan(ratio_ctot_box)
    ct_start = float(ratio_ctot_box[valid_ct][0]) if valid_ct.any() else np.nan
    ct_end   = float(ratio_ctot_box[valid_ct][-1]) if valid_ct.any() else np.nan

    valid_at = ~np.isnan(ratio_atc_box)
    at_start = float(ratio_atc_box[valid_at][0]) if valid_at.any() else np.nan

    thr_bin = None
    thr_mask = valid_at & (ratio_atc_box < 10)
    if thr_mask.any():
        thr_bin = int(bins_arr[thr_mask][0])

    lines = []
    lines.append("Datenbasis (Anteil Flüge)")
    lines.append("──────────────────────")

    if not np.isnan(ct_start):
        lines.append(f"CTOT vorhanden bei {ct_start:.0f}%")
        lines.append("der Flüge bei ATOT")
    if not np.isnan(ct_end):
        lines.append(f"→ ca. {ct_end:.0f}% der Flüge")
        lines.append(f"bei {int(time_max)} min vor ATOT")

    lines.append("")

    if not np.isnan(at_start):
        lines.append(f"ATC-TTOT vorhanden bei {at_start:.0f}%")
        lines.append("der Flüge bei ATOT")
        if thr_bin is not None:
            lines.append(f"→ ab ca. {thr_bin} min vor ATOT")
            lines.append("keine verwertbaren ATC-TTOT mehr")

    textstr = "\n".join(lines)

    props = dict(
        boxstyle="round,pad=0.6",
        facecolor="white",
        edgecolor="black",
        alpha=0.85
    )

    ax1.text(
        0.98, 0.05,
        textstr,
        transform=ax1.transAxes,
        fontsize=11,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=props,
    )

    ax1.set_title(f"Mean-Verläufe (geglättet, Count ≥ {MIN_COUNT})", fontsize=14)
    ax1.set_xlabel("Min bis ATOT (Bin)")
    ax1.set_ylabel("Delta (min)")
    ax1.set_xlim(TIME_MIN, time_max)
    ax1.set_xticks(range(TIME_MIN, int(time_max) + 1, 20))
    ax1.set_ylim(bottom=0)
    ax1.grid(True)
    ax1.legend()

    st.pyplot(fig1)

    # ================================================================
    # PANEL 2 — Stabilität innerhalb ±window
    # ================================================================
    st.subheader("Panel 2 – Stabilität innerhalb eines Δ-Fensters")

    window = st.slider(
        "Δ-Fenster für Stabilität (± Minuten)",
        min_value=1,
        max_value=15,
        value=3,
        step=1
    )

    col_p2_1, col_p2_2, col_p2_3 = st.columns(3)
    with col_p2_1:
        show_p2_etot = st.checkbox("ETOT in Panel 2", value=False)
    with col_p2_2:
        show_p2_ctot = st.checkbox("CTOT in Panel 2", value=False)
    with col_p2_3:
        show_p2_atc = st.checkbox("ATC-TTOT in Panel 2", value=False)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    bins = etot_stats.index

    pct_etot = percent_within_window(df, bins, "Delta - ETOT (min)", window, DELTA_LIMIT)
    pct_ctot = percent_within_window(df, bins, "Delta - CTOT (min)", window, DELTA_LIMIT)
    pct_atc  = percent_within_window(df, bins, "Delta - ATC TTOT (min)", window, DELTA_LIMIT)

    # ATC-TTOT nur bis Minute 25 anzeigen
    pct_atc_filtered = pct_atc.copy()
    pct_atc_filtered[bins > 25] = np.nan

    if show_p2_etot:
        ax2.plot(
            bins, pct_etot,
            marker="o", linewidth=2, color=colors["etot"],
            label=f"ETOT ±{window} min"
        )

    if show_p2_ctot:
        ax2.plot(
            bins, pct_ctot,
            marker="o", linewidth=2, color=colors["ctot"],
            label=f"CTOT ±{window} min"
        )

    if show_p2_atc:
        ax2.plot(
            bins, pct_atc_filtered,
            marker="o", linewidth=2, color=colors["atc"],
            label=f"ATC-TTOT ±{window} min (bis 25 min)"
        )

    ax2.set_title(f"Stabilität innerhalb ±{window} Minuten Delta", fontsize=14)
    ax2.set_xlabel("Min bis ATOT (Bin)")
    ax2.set_ylabel("Anteil der Flüge (%)")
    ax2.set_xlim(TIME_MIN, time_max)
    ax2.set_xticks(range(TIME_MIN, int(time_max) + 1, 20))
    ax2.set_ylim(0, 100)
    ax2.grid(True)
    ax2.legend()

    st.pyplot(fig2)

    # ================================================================
    # PANEL 3 — Airline-Kategorien
    # ================================================================
    st.subheader("Panel 3 – Mean Delta ETOT je Airline-Kategorie")

    # Checkboxes für Kategorien
    p3_cols = st.columns(len(CATEGORIES_OF_INTEREST))
    show_cat = {}
    for i, cat in enumerate(CATEGORIES_OF_INTEREST):
        with p3_cols[i]:
            show_cat[cat] = st.checkbox(cat, value=False, key=f"p3_{cat}")

    fig3, ax3 = plt.subplots(figsize=(10, 5))

    cat_grp = df_etot.groupby(["bin", "AirlineCategory"])["Delta - ETOT (min)"].agg(
        mean="mean",
        count="count"
    )

    category_colors = sns.color_palette("tab10", len(CATEGORIES_OF_INTEREST))

    for i, cat in enumerate(CATEGORIES_OF_INTEREST):
        if not show_cat.get(cat, False):
            continue
        if cat not in cat_grp.index.get_level_values("AirlineCategory"):
            continue

        stats_cat = cat_grp.xs(cat, level="AirlineCategory").sort_index()
        x_vals = stats_cat.index.values
        y_vals = stats_cat["mean"].values

        if len(y_vals) >= 5:
            y_smooth = savgol_filter(y_vals, 5, 2)
        else:
            y_smooth = y_vals

        # Count nur aus bin 0
        n0 = int(stats_cat.loc[stats_cat.index == 0, "count"].sum())

        ax3.plot(
            x_vals,
            y_smooth,
            marker="o",
            linewidth=2,
            label=f"{cat} (n={n0})",
            color=category_colors[i]
        )

    ax3.set_xlim(TIME_MIN, time_max)
    ax3.set_xticks(range(TIME_MIN, int(time_max) + 1, 20))
    ax3.set_xlabel("Min bis ATOT (Bin)")
    ax3.set_ylabel("Mean Delta ETOT (min)")
    ax3.set_title("Mean Delta ETOT je Airline-Kategorie über Zeit", fontsize=14)
    ax3.grid(True)
    ax3.legend()

    st.pyplot(fig3)

    # ================================================================
    # PANEL 4 — Runways
    # ================================================================
    st.subheader("Panel 4 – Mean Delta ETOT je Runway")

    p4_cols = st.columns(len(RUNWAYS_OF_INTEREST))
    show_rw = {}
    for i, rw in enumerate(RUNWAYS_OF_INTEREST):
        with p4_cols[i]:
            show_rw[rw] = st.checkbox(f"RWY {rw}", value=False, key=f"p4_{rw}")

    fig4, ax4 = plt.subplots(figsize=(10, 5))

    runway_grp = df_etot.groupby(["bin", "Runway"])["Delta - ETOT (min)"].agg(
        mean="mean",
        count="count"
    )

    runway_colors = sns.color_palette("Set2", len(RUNWAYS_OF_INTEREST))

    for i, rw in enumerate(RUNWAYS_OF_INTEREST):
        if not show_rw.get(rw, False):
            continue
        if rw not in runway_grp.index.get_level_values("Runway"):
            continue

        stats_rw = runway_grp.xs(rw, level="Runway").sort_index()
        x_vals = stats_rw.index.values
        y_vals = stats_rw["mean"].values

        if len(y_vals) >= 5:
            y_smooth = savgol_filter(y_vals, 5, 2)
        else:
            y_smooth = y_vals

        n0 = int(stats_rw.loc[stats_rw.index == 0, "count"].sum())

        ax4.plot(
            x_vals,
            y_smooth,
            marker="o",
            linewidth=2,
            label=f"RWY {rw} (n={n0})",
            color=runway_colors[i]
        )

    ax4.set_xlim(TIME_MIN, time_max)
    ax4.set_xticks(range(TIME_MIN, int(time_max) + 1, 20))
    ax4.set_xlabel("Min bis ATOT (Bin)")
    ax4.set_ylabel("Mean Delta ETOT (min)")
    ax4.set_title("Mean Delta ETOT je Runway über Zeit", fontsize=14)
    ax4.grid(True)
    ax4.legend()

    st.pyplot(fig4)

    # ================================================================
    # Excel-Summary Download
    # ================================================================
    summary = pd.DataFrame({
        "bin": etot_stats.index,
        "ETOT_mean": etot_stats["mean"],
        "ETOT_count": etot_stats["count"],
        "CTOT_mean": ctot_stats["mean"].reindex(etot_stats.index),
        "CTOT_count": ctot_stats["count"].reindex(etot_stats.index),
        "ATC_mean":  atc_stats["mean"].reindex(etot_stats.index),
        "ATC_count": atc_stats["count"].reindex(etot_stats.index),
        "CTOT_ETOT_ratio_%": ratio_ctot,
        "ATC_ETOT_ratio_%":  ratio_atc,
    })

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        summary.to_excel(writer, index=False, sheet_name="Summary")
    output.seek(0)

    st.download_button(
        "Excel-Summary herunterladen",
        data=output.getvalue(),
        file_name="auswertung_bins_streamlit.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    main()
