import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io
import requests
import base64
import datetime
import plotly.graph_objects as go

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

HIST_LIMIT = 30

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
    return (
        sub.groupby("bin")[col]
        .agg(mean="mean", std="std", count="count")
        .sort_index()
    )


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
# Plotly Helpers
# ================================================================
def hex_to_rgba(hex_color: str, alpha: float = 0.10) -> str:
    """'#RRGGBB' -> 'rgba(r,g,b,a)'"""
    hex_color = hex_color.strip().lstrip("#")
    if len(hex_color) != 6:
        return f"rgba(0,0,0,{alpha})"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _plotly_layout(compact, x_title, y_title, height_big=520, height_small=380):
    return dict(
        template="plotly_white",
        height=height_small if compact else height_big,
        margin=dict(l=20, r=20, t=20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title=x_title,
        yaxis_title=y_title,
        hovermode="x unified",
    )


def add_mean_series(fig, stats, label, color, min_count, show_std):
    valid = stats["count"] >= min_count
    if valid.sum() == 0:
        return

    x = stats.index[valid].to_numpy()
    m = smooth(stats.loc[valid, "mean"]).to_numpy()
    sd = smooth(stats.loc[valid, "std"]).to_numpy()
    n = stats.loc[valid, "count"].to_numpy()

    customdata = np.column_stack([sd, n])

    fig.add_trace(
        go.Scatter(
            x=x,
            y=m,
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=2),
            marker=dict(size=6),
            customdata=customdata,
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Bin: %{x} min<br>"
                "Mean: %{y:.2f} min<br>"
                "Std: %{customdata[0]:.2f} min<br>"
                "n: %{customdata[1]}<extra></extra>"
            ),
        )
    )

    if show_std:
        upper = m + sd
        lower = m - sd

        fill_rgba = hex_to_rgba(color, alpha=0.10)

        fig.add_trace(
            go.Scatter(
                x=x,
                y=upper,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=lower,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=fill_rgba,
                showlegend=False,
                hoverinfo="skip",
            )
        )


# ================================================================
# MAIN APP
# ================================================================
def main():
    if not check_password():
        return

    # ------------------ Compact Toggle (ohne Sidebar) ------------------
    top_left, _ = st.columns([1, 2])
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
    # TAB 1 – Mean-Verläufe (Plotly Hover)
    # ================================================================
    with tab1:
        st.markdown('<div class="acg-panel">', unsafe_allow_html=True)
        st.subheader("Panel 1 – Mean-Verläufe ETOT / CTOT / ATC-TTOT (Hover)")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            s_etot = st.checkbox("ETOT anzeigen", key="p1_etot")
        with c2:
            s_ctot = st.checkbox("CTOT anzeigen", key="p1_ctot")
        with c3:
            s_atc = st.checkbox("ATC-TTOT anzeigen", key="p1_atc")
        with c4:
            show_std = st.checkbox("±1σ anzeigen", True, key="p1_std")

        fig = go.Figure()

        if s_etot:
            add_mean_series(fig, etot_stats, "ETOT (Abs)", colors["etot"], MIN_COUNT, show_std)
        if s_ctot:
            add_mean_series(fig, ctot_stats, "CTOT (Abs)", colors["ctot"], MIN_COUNT, show_std)
        if s_atc:
            add_mean_series(fig, atc_stats, "ATC-TTOT (Abs)", colors["atc"], MIN_COUNT, show_std)

        fig.update_layout(**_plotly_layout(compact, "Min vor ATOT", "Delta (min)"))
        st.plotly_chart(fig, use_container_width=True, key="tab1_plot")
        st.markdown("</div>", unsafe_allow_html=True)

    # ================================================================
    # TAB 2 – Stabilität (Plotly Hover)
    # ================================================================
    with tab2:
        st.markdown('<div class="acg-panel">', unsafe_allow_html=True)
        st.subheader("Panel 2 – Stabilität (± Zeitfenster) (Hover)")

        window = st.slider("Fenster (± Minuten)", 1, 15, 3, key="p2_window")
        bins = etot_stats.index.to_numpy()

        pct_etot = percent_within_window(df, etot_stats.index, COL_ETOT, window, DELTA_LIMIT)
        pct_ctot = percent_within_window(df, etot_stats.index, COL_CTOT, window, DELTA_LIMIT)
        pct_atc = percent_within_window(df, etot_stats.index, COL_ATC, window, DELTA_LIMIT)

        fig = go.Figure()

        def add_pct(name, y, color, n0):
            fig.add_trace(
                go.Scatter(
                    x=bins,
                    y=y,
                    mode="lines+markers",
                    name=f"{name} ±{window} min (n={n0})",
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"
                        "Bin: %{x} min<br>"
                        "Anteil: %{y:.1f}%<extra></extra>"
                    ),
                )
            )

        add_pct("ETOT", pct_etot, colors["etot"], n0_etot)
        add_pct("CTOT", pct_ctot, colors["ctot"], n0_ctot)
        add_pct("ATC-TTOT", pct_atc, colors["atc"], n0_atc)

        fig.update_yaxes(range=[0, 100])
        fig.update_layout(**_plotly_layout(compact, "Min vor ATOT", "Anteil (%)"))
        st.plotly_chart(fig, use_container_width=True, key="tab2_plot")
        st.markdown("</div>", unsafe_allow_html=True)

    # ================================================================
    # TAB 3 – Airline Kategorien (Plotly Hover) + Hinweis wenn nichts ausgewählt
    # ================================================================
    with tab3:
        st.markdown('<div class="acg-panel">', unsafe_allow_html=True)
        st.subheader("Panel 3 – Airline-Kategorien (ETOT) (Hover)")

        cols = st.columns(len(CATEGORIES_OF_INTEREST))
        show_cat = {}
        for i, cat in enumerate(CATEGORIES_OF_INTEREST):
            with cols[i]:
                show_cat[cat] = st.checkbox(cat, False, key=f"p3_cat_{cat}")

        cat_grp = df_etot.groupby(["bin", "AirlineCategory"])[COL_ETOT].mean()
        fig = go.Figure()

        any_selected = False
        for cat in CATEGORIES_OF_INTEREST:
            if not show_cat.get(cat, False):
                continue
            if cat not in cat_grp.index.get_level_values(1):
                continue

            any_selected = True
            series = cat_grp.xs(cat, level="AirlineCategory").sort_index()
            x = series.index.to_numpy()
            y = smooth(series).to_numpy()
            n0_cat = int(cat_counts0.get(cat, 0))

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers",
                    name=f"{cat} (n={n0_cat})",
                    line=dict(width=2),
                    marker=dict(size=6),
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"
                        "Bin: %{x} min<br>"
                        "Mean ΔETOT: %{y:.2f} min<extra></extra>"
                    ),
                )
            )

        fig.update_layout(**_plotly_layout(compact, "Min vor ATOT", "Delta ETOT (min)"))

        if not any_selected:
            fig.add_annotation(
                text="Bitte mindestens eine Airline-Kategorie auswählen.",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color="gray"),
            )

        st.plotly_chart(fig, use_container_width=True, key="tab3_plot")
        st.markdown("</div>", unsafe_allow_html=True)

    # ================================================================
    # TAB 4 – Runways (Plotly Hover) + Hinweis wenn nichts ausgewählt
    # ================================================================
    with tab4:
        st.markdown('<div class="acg-panel">', unsafe_allow_html=True)
        st.subheader("Panel 4 – Runways (ETOT) (Hover)")

        cols = st.columns(len(RUNWAYS_OF_INTEREST))
        show_rw = {}
        for i, rw in enumerate(RUNWAYS_OF_INTEREST):
            with cols[i]:
                show_rw[rw] = st.checkbox(f"RWY {rw}", False, key=f"p4_rw_{rw}")

        rw_grp = df_etot.groupby(["bin", "Runway"])[COL_ETOT].mean()
        fig = go.Figure()

        any_selected = False
        for rw in RUNWAYS_OF_INTEREST:
            if not show_rw.get(rw, False):
                continue
            if rw not in rw_grp.index.get_level_values(1):
                continue

            any_selected = True
            series = rw_grp.xs(rw, level="Runway").sort_index()
            x = series.index.to_numpy()
            y = smooth(series).to_numpy()
            n0_rw = int(rw_counts0.get(rw, 0))

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers",
                    name=f"RWY {rw} (n={n0_rw})",
                    line=dict(width=2),
                    marker=dict(size=6),
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"
                        "Bin: %{x} min<br>"
                        "Mean ΔETOT: %{y:.2f} min<extra></extra>"
                    ),
                )
            )

        fig.update_layout(**_plotly_layout(compact, "Min vor ATOT", "Delta ETOT (min)"))

        if not any_selected:
            fig.add_annotation(
                text="Bitte mindestens eine Runway auswählen.",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color="gray"),
            )

        st.plotly_chart(fig, use_container_width=True, key="tab4_plot")
        st.markdown("</div>", unsafe_allow_html=True)

    # ================================================================
    # TAB 5 – Histogramm Signed (Plotly Hover)
    # ================================================================
    with tab5:
        st.markdown('<div class="acg-panel">', unsafe_allow_html=True)
        st.subheader("Panel 5 – Histogramm (DeltaSigned) (Hover)")

        c1, c2, c3 = st.columns(3)
        with c1:
            h_etot = st.checkbox("ETOT (Signed) anzeigen", True, key="p5_etot")
        with c2:
            h_ctot = st.checkbox("CTOT (Signed) anzeigen", True, key="p5_ctot")
        with c3:
            h_atc = st.checkbox("ATC-TTOT (Signed) anzeigen", False, key="p5_atc")

        show_mean_bias = st.checkbox("Mean Bias anzeigen (vertikale Linie)", True, key="p5_meanbias")
        hist_bin_width = st.slider("Histogramm-Breite (min)", 1, 10, 2, step=1, key="p5_bw")

        fig = go.Figure()

        def add_hist(col, name, color):
            s = df[col].dropna()
            s = s[(s >= -HIST_LIMIT) & (s <= HIST_LIMIT)]
            if s.empty:
                return

            low, high = np.percentile(s, [1, 99])
            s = s[(s >= low) & (s <= high)]
            if s.empty:
                return

            mu = float(s.mean())

            fig.add_trace(
                go.Histogram(
                    x=s,
                    xbins=dict(start=-HIST_LIMIT, end=HIST_LIMIT, size=hist_bin_width),
                    histnorm="probability density",
                    name=f"{name} (n={len(s)}, μ={mu:.2f})" if show_mean_bias else f"{name} (n={len(s)})",
                    opacity=0.45,
                    marker=dict(color=color),
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"
                        "Bin: %{x} min<br>"
                        "Dichte: %{y:.4f}<extra></extra>"
                    ),
                )
            )

            if show_mean_bias:
                fig.add_vline(x=mu, line_width=2, line_dash="dash", line_color=color)

        if h_etot:
            add_hist(COL_ETOT_S, "ETOT", colors["etot"])
        if h_ctot:
            add_hist(COL_CTOT_S, "CTOT", colors["ctot"])
        if h_atc:
            add_hist(COL_ATC_S, "ATC-TTOT", colors["atc"])

        cap_left, cap_mid, cap_right = st.columns([1, 2, 1])
        with cap_left:
            st.caption("⬅️ **zu früh gestartet** (positiver Delta)")
        with cap_mid:
            st.caption("⏱️ Referenz: 0 = pünktlich")
        with cap_right:
            st.caption("➡️ **zu spät gestartet** (negativer Delta)")

        fig.update_layout(**_plotly_layout(compact, "DeltaSigned (min)  ⟵ zu früh | zu spät ⟶", "Dichte", height_big=520, height_small=420))
        fig.update_xaxes(autorange="reversed", range=[HIST_LIMIT, -HIST_LIMIT])
        fig.add_vline(x=0, line_width=1, line_color="black")

        st.plotly_chart(fig, use_container_width=True, key="tab5_plot")
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
            "ETOT_std": etot_stats["std"],
            "ETOT_count": etot_stats["count"],
            "CTOT_mean": ctot_stats["mean"].reindex(etot_stats.index),
            "CTOT_std": ctot_stats["std"].reindex(etot_stats.index),
            "CTOT_count": ctot_stats["count"].reindex(etot_stats.index),
            "ATC_mean": atc_stats["mean"].reindex(etot_stats.index),
            "ATC_std": atc_stats["std"].reindex(etot_stats.index),
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
