# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Bond & Spread Trade Manager", layout="wide")

st.title("Bond & Bond-Spread Viewer")

st.caption(
    "Upload two time series (e.g., bond yields, prices, or spreads). "
    "We’ll align them by date, chart each, compute (1)-(2), and produce a z-score."
)

# -----------------------------
# Helpers
# -----------------------------
def read_timeseries_csv(file: io.BytesIO) -> pd.DataFrame:
    """
    Expect CSV with:
    - Column 1: dates in dd/mm/yyyy (used as index).
    - Column 2: text (ignored).
    - Column 3: numeric price (returned as the single series).
    """
    file.seek(0)
    try:
        df = pd.read_csv(file)
    except ValueError:
        return pd.DataFrame()

    if df.shape[1] < 3:
        return pd.DataFrame()

    # Column 1: dd/mm/yyyy dates, Column 3: price; column 2 ignored.
    date_raw = df.iloc[:, 0].astype(str).str.strip()
    price_raw = df.iloc[:, 2]

    date_index = pd.to_datetime(date_raw, format="%d/%m/%Y", errors="coerce")
    price = pd.to_numeric(price_raw, errors="coerce")

    price_col = df.columns[2]
    ts = pd.DataFrame({price_col: price})
    ts.index = date_index
    ts = ts[~ts.index.isna()]
    ts = ts.dropna(subset=[price_col])
    ts = ts.sort_index()
    return ts


def ensure_single_series(df: pd.DataFrame, label_prefix: str) -> tuple[pd.Series, str]:
    """
    If multiple numeric columns exist, let the user choose which one is the series.
    Returns (series, chosen_column_name).
    """
    if df is None or df.empty:
        return None, None

    cols = df.columns.tolist()
    if len(cols) == 0:
        return None, None

    chosen = st.selectbox(
        f"{label_prefix}: choose value column",
        cols,
        index=0,
        key=f"sel_{label_prefix}",
    )
    s = df[chosen].astype(float)
    s.name = chosen
    return s, chosen


def align_two(s1: pd.Series, s2: pd.Series, how="inner") -> pd.DataFrame:
    df = pd.concat({"series1": s1, "series2": s2}, axis=1).dropna(how="any")
    return df


def rolling_zscore(x: pd.Series, window: int) -> pd.Series:
    mu = x.rolling(window).mean()
    sd = x.rolling(window).std(ddof=1)
    return (x - mu) / sd


def spread_vol(spread: pd.Series, window: int, diff_first: bool, annualize: bool) -> float:
    """
    Compute volatility of the spread:
    - If diff_first=True, use daily changes of spread (good for yields, bp).
    - Else use level volatility (less common for spreads).
    """
    if diff_first:
        x = spread.diff()
    else:
        x = spread

    vol = x.rolling(window).std(ddof=1).iloc[-1]
    if pd.isna(vol):
        return np.nan
    if annualize:
        vol *= np.sqrt(252.0)
    return float(vol)


# -----------------------------
# Sidebar (file selection boxes)
# -----------------------------
with st.sidebar:
    st.header("Upload CSVs")

    file1 = st.file_uploader("Upload 1 (.csv)", type=["csv"], key="f1")
    load1 = st.button("LOAD 1")
    file2 = st.file_uploader("Upload 2 (.csv)", type=["csv"], key="f2")
    load2 = st.button("LOAD 2")

    st.markdown("---")
    st.subheader("Analytics Settings")
    roll_spread = st.number_input("Z-Score rolling window (days)", min_value=5, max_value=2520, value=60, step=5)
    roll_vol = st.number_input("Vol window (days)", min_value=5, max_value=2520, value=60, step=5)
    roll_corr = st.number_input("Correlation window (days)", min_value=5, max_value=2520, value=60, step=5)
    diff_first = st.checkbox("Use daily Δ for vol (recommended)", value=True)
    annualize = st.checkbox("Annualize vol (×√252)", value=False)

# -----------------------------
# Load data
# -----------------------------
if "df1" not in st.session_state:
    st.session_state.df1 = None
if "df2" not in st.session_state:
    st.session_state.df2 = None

if load1 and file1 is not None:
    st.session_state.df1 = read_timeseries_csv(file1)

if load2 and file2 is not None:
    st.session_state.df2 = read_timeseries_csv(file2)

df1 = st.session_state.df1
df2 = st.session_state.df2

# -----------------------------
# Main layout according to sketch
# -----------------------------
col_top_l, col_top_r = st.columns(2, gap="large")

with col_top_l:
    st.subheader("Chart 1 (Time Series)")
    if df1 is None:
        st.info("Upload 1 and click LOAD 1 to display Chart 1.")
        s1 = None
        c1 = None
    else:
        s1, c1 = ensure_single_series(df1, "Upload 1")
        if s1 is not None and not s1.empty:
            st.line_chart(s1, height=260)

with col_top_r:
    st.subheader("Chart 2")
    if df2 is None:
        st.info("Upload 2 and click LOAD 2 to display Chart 2.")
        s2 = None
        c2 = None
    else:
        s2, c2 = ensure_single_series(df2, "Upload 2")
        if s2 is not None and not s2.empty:
            st.line_chart(s2, height=260)

st.markdown("---")

col_mid_l, col_mid_r = st.columns(2, gap="large")

with col_mid_l:
    st.subheader("Spread (1) − (2)")
    spread = None
    merged = None
    if s1 is None or s2 is None:
        st.info("Load both series above to compute the spread.")
    else:
        merged = align_two(s1, s2, how="inner")
        if merged.empty:
            st.warning("No overlapping dates between the two series.")
        else:
            spread = merged["series1"] - merged["series2"]
            spread.name = f"{c1 or 'series1'} - {c2 or 'series2'}"
            st.line_chart(spread, height=260)

with col_mid_r:
    st.subheader("Z-Score ( (1) − (2) )")
    if spread is None or spread.empty:
        st.info("Spread required to compute Z-Score.")
        z = None
    else:
        z = rolling_zscore(spread, int(roll_spread))
        st.line_chart(z, height=260)

st.markdown("---")

st.subheader("Rolling Correlation")
if merged is None or merged.empty:
    st.info("Load both series with overlapping dates to compute correlation.")
    corr = None
else:
    corr = merged["series1"].rolling(int(roll_corr)).corr(merged["series2"])
    corr_df = corr.to_frame(name="correlation")
    st.line_chart(corr_df, height=260)

st.markdown("---")

# Bottom KPIs (Spread Vol & Spread Z-Score)
kpi_l, kpi_r = st.columns(2, gap="large")

with kpi_l:
    st.markdown("#### Spread Vol")
    if spread is None or spread.empty:
        st.info("Load both series and compute the spread first.")
    else:
        vol_value = spread_vol(spread, int(roll_vol), diff_first=diff_first, annualize=annualize)
        if np.isnan(vol_value):
            st.warning("Not enough data in the chosen window to compute volatility.")
        else:
            unit_hint = "(annualized)" if annualize else "(rolling)"
            st.metric(label=f"Std Dev {unit_hint}", value=f"{vol_value:,.6f}")

with kpi_r:
    st.markdown("#### Spread Z-Score (Latest)")
    if z is None or z.dropna().empty:
        st.info("Z-Score not available.")
    else:
        latest_z = float(z.dropna().iloc[-1])
        st.metric(label="Latest Z", value=f"{latest_z:,.3f}")

# Optional downloads
st.markdown("---")
dl_cols = st.columns(3)
with dl_cols[0]:
    if s1 is not None:
        st.download_button(
            "Download Series 1 (CSV)",
            s1.to_csv().encode("utf-8"),
            file_name="series1.csv",
            mime="text/csv",
        )
with dl_cols[1]:
    if s2 is not None:
        st.download_button(
            "Download Series 2 (CSV)",
            s2.to_csv().encode("utf-8"),
            file_name="series2.csv",
            mime="text/csv",
        )
with dl_cols[2]:
    if spread is not None:
        out = pd.DataFrame({"spread": spread})
        if "z" in locals() and z is not None:
            out["zscore"] = z
        if "corr" in locals() and corr is not None:
            out["correlation"] = corr
        st.download_button(
            "Download Spread, Z & Corr (CSV)",
            out.to_csv().encode("utf-8"),
            file_name="spread_z_corr.csv",
            mime="text/csv",
        )

# Footer notes
st.caption(
    "Tips: Use consistent date columns in your CSVs. You can toggle whether spread vol uses daily deltas "
    "or levels, and whether it is annualized. Z-Score is calculated on spread levels over a rolling window."
)
