# app.py
import io
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from pandas.tseries.offsets import BDay, DateOffset

from trade_sizing import pnl_series, rolling_vol, spread_returns_bps
from carry_rolldown import leg_carry_roll

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


def render_line_chart(
    data: pd.Series | pd.DataFrame,
    *,
    height: int = 260,
    y_label: str | None = None,
    title: str | None = None,
    upper_threshold: float | None = None,
    lower_threshold: float | None = None,
    upper_color: str = "#2ca02c",
    lower_color: str = "#d62728",
    default_color: str = "#1f77b4",
    dropna: bool = True,
) -> None:
    """Render a line chart sized to the data using Altair."""

    if data is None:
        return

    if isinstance(data, pd.Series):
        series = data.dropna() if dropna else data.copy()
        if series.empty:
            return
        value_col = series.name or "value"
        if not isinstance(value_col, str):
            value_col = str(value_col)
        df = series.reset_index()
        df.columns = [df.columns[0], value_col]
    else:
        df = data.dropna(how="any") if dropna else data.copy()
        if df.empty:
            return
        df = df.reset_index()
        trailing_cols = [c for c in df.columns[1:] if c not in ("level_0", "level_1")]
        value_col = (trailing_cols[-1] if trailing_cols else df.columns[-1]) or "value"
        if not isinstance(value_col, str):
            value_col = str(value_col)
        df = df.rename(columns={df.columns[-1]: value_col})

    x_col = df.columns[0]
    if not pd.api.types.is_datetime64_any_dtype(df[x_col]):
        df[x_col] = pd.to_datetime(df[x_col], errors="coerce")
        df = df.dropna(subset=[x_col])
        if df.empty:
            return

    encoding = {
        "x": alt.X(f"{x_col}:T", title="Date"),
        "y": alt.Y(
            f"{value_col}:Q",
            title=y_label or value_col,
            scale=alt.Scale(zero=False),
        ),
    }

    layers = [
        alt.Chart(df).mark_line(color=default_color).encode(**encoding)
    ]

    if upper_threshold is not None:
        df_upper = df[[x_col, value_col]].copy()
        df_upper.loc[df_upper[value_col] <= upper_threshold, value_col] = np.nan
        layers.append(
            alt.Chart(df_upper)
            .mark_line(color=upper_color)
            .encode(**encoding)
        )

    if lower_threshold is not None:
        df_lower = df[[x_col, value_col]].copy()
        df_lower.loc[df_lower[value_col] >= lower_threshold, value_col] = np.nan
        layers.append(
            alt.Chart(df_lower)
            .mark_line(color=lower_color)
            .encode(**encoding)
        )

    chart = alt.layer(*layers).properties(height=height)

    if title:
        chart = chart.properties(title=title)

    st.altair_chart(chart, use_container_width=True)


def last_valid_value(series: pd.Series | None) -> float:
    if series is None:
        return float("nan")
    ser = series.dropna()
    if ser.empty:
        return float("nan")
    return float(ser.iloc[-1])


def normalize_yield(value: float | None) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    try:
        val = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if np.isnan(val):
        return float("nan")
    if abs(val) > 1.0:
        return val / 100.0
    return val


def format_compact(value: float | None, *, decimals: int = 2, unit: str = "") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    sign = "-" if value < 0 else ""
    abs_val = abs(float(value))
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs_val >= threshold:
            scaled = abs_val / threshold
            return f"{sign}{unit}{scaled:.{decimals}f}{suffix}"
    return f"{sign}{unit}{abs_val:,.{decimals}f}"


def compute_duration_metrics(
    *,
    yield_rate: float,
    coupon_pct: float,
    price: float,
    frequency: int,
    settlement: pd.Timestamp,
    maturity: pd.Timestamp,
    day_count: float = 365.25,
) -> tuple[float, float, float]:
    if (
        pd.isna(yield_rate)
        or pd.isna(coupon_pct)
        or pd.isna(price)
        or frequency <= 0
        or price <= 0
    ):
        return float("nan"), float("nan"), float("nan")

    settlement = pd.Timestamp(settlement)
    maturity = pd.Timestamp(maturity)
    if settlement >= maturity:
        return float("nan"), float("nan"), float("nan")

    coupon_rate = float(coupon_pct) / 100.0
    freq = int(frequency)
    y = float(yield_rate)

    step_months = 12 // freq
    if step_months <= 0:
        return float("nan"), float("nan"), float("nan")

    coupon_dates: list[pd.Timestamp] = []
    current = maturity
    while current > settlement:
        coupon_dates.append(current)
        current -= DateOffset(months=step_months)

    if not coupon_dates:
        return float("nan"), float("nan"), float("nan")

    coupon_dates.sort()
    times = np.array([(date - settlement).days / day_count for date in coupon_dates], dtype=float)
    if times.size == 0:
        return float("nan"), float("nan"), float("nan")

    if (1 + y / freq) <= 0:
        return float("nan"), float("nan"), float("nan")

    coupon_payment = coupon_rate / freq * 100.0
    cashflows = np.full(times.shape, coupon_payment)
    cashflows[-1] += 100.0

    discounts = (1 + y / freq) ** (freq * times)
    pv = cashflows / discounts
    pv_sum = pv.sum()
    if pv_sum <= 0:
        return float("nan"), float("nan"), float("nan")

    macaulay = float((times * pv).sum() / pv_sum)
    modified = macaulay / (1 + y / freq)
    dv01_cents = modified * price * 0.0001 * 100.0

    return macaulay, modified, dv01_cents


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

    today = pd.Timestamp.today().normalize()
    default_settlement_ts = today + BDay(2)
    default_maturity_ts = default_settlement_ts + DateOffset(years=10)

    settlement_date = st.date_input(
        "Settlement Date",
        value=default_settlement_ts.date(),
    )
    coupon_frequency = st.selectbox(
        "Coupon Frequency",
        ["Annual", "Semi-Annual"],
        index=0,
    )

    st.markdown("**Leg Inputs**")
    col_leg1, col_leg2 = st.columns(2, gap="small")
    with col_leg1:
        coupon_leg1 = st.number_input(
            "Coupon Leg 1 (%)",
            value=4.0,
            step=0.1,
            format="%0.2f",
        )
        repo_leg1 = st.number_input(
            "Repo Rate Leg 1 (%)",
            value=2.1,
            step=0.1,
            format="%0.2f",
        )
        roll_down_leg1 = st.number_input(
            "Roll Down Leg 1 (bp)",
            value=10.0,
            step=1.0,
            format="%0.1f",
        )
        price_leg1 = st.number_input(
            "Gross Price Leg 1",
            min_value=0.0,
            value=104.0,
            step=0.1,
            format="%0.2f",
        )
        maturity_leg1 = st.date_input(
            "Maturity Leg 1",
            value=default_maturity_ts.date(),
        )
    with col_leg2:
        coupon_leg2 = st.number_input(
            "Coupon Leg 2 (%)",
            value=3.5,
            step=0.1,
            format="%0.2f",
        )
        repo_leg2 = st.number_input(
            "Repo Rate Leg 2 (%)",
            value=2.0,
            step=0.1,
            format="%0.2f",
        )
        roll_down_leg2 = st.number_input(
            "Roll Down Leg 2 (bp)",
            value=10.0,
            step=1.0,
            format="%0.1f",
        )
        price_leg2 = st.number_input(
            "Gross Price Leg 2",
            min_value=0.0,
            value=100.0,
            step=0.1,
            format="%0.2f",
        )
        maturity_leg2 = st.date_input(
            "Maturity Leg 2",
            value=default_maturity_ts.date(),
        )
    target_daily_vol = st.number_input("Trade daily vol target (EUR)", min_value=0.0, value=20000.0, step=100.0)

# -----------------------------
# Load data
# -----------------------------
if "df1" not in st.session_state:
    st.session_state.df1 = None
if "df2" not in st.session_state:
    st.session_state.df2 = None
if "file1_name" not in st.session_state:
    st.session_state.file1_name = None
if "file2_name" not in st.session_state:
    st.session_state.file2_name = None
if "defaults_loaded" not in st.session_state:
    st.session_state.defaults_loaded = False

if not st.session_state.defaults_loaded:
    base_dir = Path(__file__).resolve().parent
    default_map = {
        "df1": "BTPS10Y.csv",
        "df2": "FRTR10Y.csv",
    }
    for key, filename in default_map.items():
        csv_path = base_dir / filename
        if csv_path.exists():
            with csv_path.open("rb") as fh:
                buffer = io.BytesIO(fh.read())
            st.session_state[key] = read_timeseries_csv(buffer)
            st.session_state[f"{key.replace('df', 'file')}_name"] = Path(filename).stem
    st.session_state.defaults_loaded = True

if load1 and file1 is not None:
    st.session_state.df1 = read_timeseries_csv(file1)
    st.session_state.file1_name = Path(file1.name).stem

if load2 and file2 is not None:
    st.session_state.df2 = read_timeseries_csv(file2)
    st.session_state.file2_name = Path(file2.name).stem

df1 = st.session_state.df1
df2 = st.session_state.df2

# -----------------------------
# Main layout according to sketch
# -----------------------------
col_top_l, col_top_r = st.columns(2, gap="large")

with col_top_l:
    chart1_title = st.session_state.file1_name or "Chart 1 (Time Series)"
    st.subheader(chart1_title)
    if df1 is None:
        st.info("Upload 1 and click LOAD 1 to display Chart 1.")
        s1 = None
        c1 = None
    else:
        s1, c1 = ensure_single_series(df1, "Upload 1")
        if s1 is not None and not s1.empty:
            render_line_chart(s1, height=260, y_label=c1)

with col_top_r:
    chart2_title = st.session_state.file2_name or "Chart 2"
    st.subheader(chart2_title)
    if df2 is None:
        st.info("Upload 2 and click LOAD 2 to display Chart 2.")
        s2 = None
        c2 = None
    else:
        s2, c2 = ensure_single_series(df2, "Upload 2")
        if s2 is not None and not s2.empty:
            render_line_chart(s2, height=260, y_label=c2)

st.markdown("---")

st.subheader("Spread Time Serie")
spread = None
merged = None
spread_returns = None
timeline_index = None
if s1 is None or s2 is None:
    st.info("Load both series above to compute the spread.")
else:
    timeline_index = s1.index.union(s2.index).sort_values()
    merged = align_two(s1, s2, how="inner")
    if merged.empty:
        st.warning("No overlapping dates between the two series.")
    else:
        spread = merged["series1"] - merged["series2"]
        spread.name = f"{c1 or 'series1'} - {c2 or 'series2'}"
        spread_chart = spread.reindex(timeline_index) if timeline_index is not None else spread
        render_line_chart(spread_chart, height=260, y_label=spread.name)

st.subheader("Rolling Z-Score Time Serie")
if spread is None or spread.empty:
    st.info("Spread required to compute Z-Score.")
    z = None
else:
    z = rolling_zscore(spread, int(roll_spread))
    z_chart = z.reindex(timeline_index) if timeline_index is not None else z
    render_line_chart(
        z_chart,
        height=260,
        y_label="Z-Score",
        upper_threshold=2.0,
        lower_threshold=-2.0,
        dropna=False,
    )

st.subheader("Rolling Spread Vol")
if spread is None or spread.empty:
    st.info("Spread required to compute rolling spread volatility.")
else:
    spread_returns = spread_returns_bps(spread)
    returns_for_vol = spread_returns.dropna()
    vol_series = returns_for_vol.rolling(int(roll_vol)).std(ddof=1)
    vol_series.name = "Rolling Spread Vol (bp)"
    vol_chart = vol_series.reindex(timeline_index) if timeline_index is not None else vol_series
    render_line_chart(vol_chart, height=260, y_label="Rolling Spread Vol (bp)", dropna=False)

st.subheader("Rolling Correlation")
if merged is None or merged.empty:
    st.info("Load both series with overlapping dates to compute correlation.")
    corr = None
else:
    corr = merged["series1"].rolling(int(roll_corr)).corr(merged["series2"])
    corr_df = corr.to_frame(name="correlation")
    corr_chart = (
        corr_df["correlation"].reindex(timeline_index) if timeline_index is not None else corr_df["correlation"]
    )
    render_line_chart(corr_chart, height=260, y_label="Correlation", dropna=False)

freq_value = 1 if coupon_frequency == "Annual" else 2
settlement_ts = pd.Timestamp(settlement_date)

yield_leg1 = normalize_yield(last_valid_value(s1))
yield_leg2 = normalize_yield(last_valid_value(s2))

macaulay_leg1, modified_leg1, dv01_leg1_cents = compute_duration_metrics(
    yield_rate=yield_leg1,
    coupon_pct=coupon_leg1,
    price=price_leg1,
    frequency=freq_value,
    settlement=settlement_ts,
    maturity=pd.Timestamp(maturity_leg1),
)

macaulay_leg2, modified_leg2, dv01_leg2_cents = compute_duration_metrics(
    yield_rate=yield_leg2,
    coupon_pct=coupon_leg2,
    price=price_leg2,
    frequency=freq_value,
    settlement=settlement_ts,
    maturity=pd.Timestamp(maturity_leg2),
)

dv01_leg1 = dv01_leg1_cents / 100.0 if not pd.isna(dv01_leg1_cents) else float("nan")
dv01_leg2 = dv01_leg2_cents / 100.0 if not pd.isna(dv01_leg2_cents) else float("nan")

duration_summary = {
    "Leg 1": {
        "yield_pct": yield_leg1 * 100.0 if not pd.isna(yield_leg1) else float("nan"),
        "macaulay": macaulay_leg1,
        "modified": modified_leg1,
        "dv01_cents": dv01_leg1_cents,
    },
    "Leg 2": {
        "yield_pct": yield_leg2 * 100.0 if not pd.isna(yield_leg2) else float("nan"),
        "macaulay": macaulay_leg2,
        "modified": modified_leg2,
        "dv01_cents": dv01_leg2_cents,
    },
}

with st.sidebar:
    st.markdown("**Computed DV01s (cents/bp)**")
    dv01_leg1_display = "n/a" if np.isnan(dv01_leg1_cents) else f"{dv01_leg1_cents:,.2f}"
    dv01_leg2_display = "n/a" if np.isnan(dv01_leg2_cents) else f"{dv01_leg2_cents:,.2f}"
    st.markdown(f"{st.session_state.file1_name or 'Leg 1'}: {dv01_leg1_display}")
    st.markdown(f"{st.session_state.file2_name or 'Leg 2'}: {dv01_leg2_display}")

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

st.markdown("---")

st.subheader("Trade Sizing")
if spread is None or spread.empty:
    st.info("Compute the spread above to enable trade sizing.")
    returns_bps = pd.Series(dtype="float64")
    pnl = pd.Series(dtype="float64")
    spread_vol_bp = float("nan")
    pnl_combined_vol = float("nan")
    leg1_vol_cents = float("nan")
    leg2_vol_cents = float("nan")
    leg1_window_vol_cents = float("nan")
    notional_leg1 = float("nan")
    notional_leg2 = float("nan")
    scaling = float("nan")
    ratio = float("nan")
    leg1_carry_cents = float("nan")
    leg2_carry_cents = float("nan")
    trade_carry_cents = float("nan")
    carry_ratio_pct = float("nan")
else:
    returns_leg1 = spread_returns_bps(s1) if s1 is not None else pd.Series(dtype="float64")
    returns_leg2 = spread_returns_bps(s2) if s2 is not None else pd.Series(dtype="float64")
    returns_df = pd.concat(
        {
            "spread": spread_returns if spread_returns is not None else pd.Series(dtype="float64"),
            "leg1": returns_leg1,
            "leg2": returns_leg2,
        },
        axis=1,
    ).dropna()

    ratio = (dv01_leg1 / dv01_leg2) if dv01_leg2 > 0 else float("nan")

    if returns_df.empty or returns_df.shape[0] < int(roll_vol):
        st.warning("Not enough overlapping data to compute trade sizing metrics.")
        returns_bps = pd.Series(dtype="float64")
        pnl = pd.Series(dtype="float64")
        spread_vol_bp = float("nan")
        pnl_combined_vol = float("nan")
        leg1_vol_cents = float("nan")
        leg2_vol_cents = float("nan")
        leg1_window_vol_cents = float("nan")
        notional_leg1 = float("nan")
        notional_leg2 = float("nan")
        scaling = float("nan")
        leg1_carry_cents = float("nan")
        leg2_carry_cents = float("nan")
        trade_carry_cents = float("nan")
        carry_ratio_pct = float("nan")
    else:
        returns_bps = returns_df["spread"]
        returns_leg1 = returns_df["leg1"]
        returns_leg2 = returns_df["leg2"]

        pnl_leg1_unit = pnl_series(returns_leg1, dv01_leg1)
        pnl_leg2_unit = pnl_series(returns_leg2, dv01_leg2)

        pnl = pnl_leg1_unit - (pnl_leg2_unit * ratio)
        pnl.name = "Daily P&L (cents)"

        spread_vol_bp = rolling_vol(returns_bps, int(roll_vol))
        pnl_combined_vol = rolling_vol(pnl, int(roll_vol))

        if (
            np.isnan(pnl_combined_vol)
            or pnl_combined_vol <= 0
            or target_daily_vol <= 0
            or np.isnan(ratio)
            or ratio <= 0
        ):
            scaling = float("nan")
        else:
            scaling = target_daily_vol / pnl_combined_vol

        if target_daily_vol <= 0 or np.isnan(dv01_leg1_cents) or dv01_leg1_cents <= 0:
            notional_leg1 = float("nan")
        else:
            notional_leg1 = (target_daily_vol * 10_000.0) / dv01_leg1_cents

        if (
            np.isnan(notional_leg1)
            or np.isnan(dv01_leg2_cents)
            or dv01_leg2_cents <= 0
        ):
            notional_leg2 = float("nan")
        else:
            notional_leg2 = notional_leg1 * (dv01_leg1_cents / dv01_leg2_cents)

        if np.isnan(spread_vol_bp):
            leg1_vol_cents = float("nan")
            leg2_vol_cents = float("nan")
            leg1_window_vol_cents = float("nan")
        else:
            leg1_vol_cents = spread_vol_bp * dv01_leg1_cents
            leg2_vol_cents = spread_vol_bp * dv01_leg2_cents
            leg1_window_vol_cents = leg1_vol_cents * np.sqrt(float(roll_vol))

        leg1_carry = leg_carry_roll(
            coupon_pct=coupon_leg1,
            repo_pct=repo_leg1,
            roll_down_bps=roll_down_leg1,
            gross_price=price_leg1,
            dv01_eur_bp=dv01_leg1,
            window_days=int(roll_vol),
        )
        leg2_carry = leg_carry_roll(
            coupon_pct=coupon_leg2,
            repo_pct=repo_leg2,
            roll_down_bps=roll_down_leg2,
            gross_price=price_leg2,
            dv01_eur_bp=dv01_leg2,
            window_days=int(roll_vol),
        )

        leg1_carry_cents = leg1_carry.total_cents
        leg2_carry_cents = leg2_carry.total_cents
        trade_carry_cents = (
            leg1_carry_cents - leg2_carry_cents * ratio if not np.isnan(ratio) else float("nan")
        )
        if (
            np.isnan(trade_carry_cents)
            or np.isnan(leg1_window_vol_cents)
            or leg1_window_vol_cents == 0
        ):
            carry_ratio_pct = float("nan")
        else:
            carry_ratio_pct = (trade_carry_cents * 100.0) / leg1_window_vol_cents

    chart_cols = st.columns(2, gap="large")
    with chart_cols[0]:
        st.write("Spread Δ (bp)")
        if returns_bps.empty:
            st.info("Not enough data to compute spread changes.")
        else:
            render_line_chart(returns_bps, height=240, y_label="Δ Spread (bp)")
    with chart_cols[1]:
        st.write("Daily P&L (cents)")
        if pnl.empty:
            st.info("Enter DV01s above to compute P&L.")
        else:
            render_line_chart(pnl, height=240, y_label="Daily P&L (cents)")

    metrics_top = st.columns(2, gap="large")
    with metrics_top[0]:
        if np.isnan(spread_vol_bp):
            st.metric("Daily Spread Vol (bp)", "n/a")
        else:
            st.metric("Daily Spread Vol (bp)", f"{spread_vol_bp:,.2f}")
    with metrics_top[1]:
        if np.isnan(pnl_combined_vol):
            st.metric("Trade Daily P&L Vol (EUR)", "n/a")
        else:
            scaled_trade_vol = (
                pnl_combined_vol * scaling if not np.isnan(scaling) else pnl_combined_vol
            )
            st.metric("Trade Daily P&L Vol (EUR)", format_compact(scaled_trade_vol, unit="€"))

    leg_metric_cols = st.columns(6, gap="large")
    with leg_metric_cols[0]:
        if np.isnan(dv01_leg1_cents):
            st.metric("Leg 1 DV01 (cents/bp)", "n/a")
        else:
            st.metric("Leg 1 DV01 (cents/bp)", f"{dv01_leg1_cents:,.2f}")
    with leg_metric_cols[1]:
        if np.isnan(leg1_vol_cents):
            st.metric("Leg 1 Daily P&L Vol (cents)", "n/a")
        else:
            st.metric("Leg 1 Daily P&L Vol (cents)", f"{leg1_vol_cents:,.2f}")
    with leg_metric_cols[2]:
        if np.isnan(notional_leg1):
            st.metric("Leg 1 Notional Needed (EUR)", "n/a")
        else:
            st.metric(
                "Leg 1 Notional Needed (EUR)",
                format_compact(abs(notional_leg1), unit="€"),
            )
    with leg_metric_cols[3]:
        if np.isnan(dv01_leg2_cents):
            st.metric("Leg 2 DV01 (cents/bp)", "n/a")
        else:
            st.metric("Leg 2 DV01 (cents/bp)", f"{dv01_leg2_cents:,.2f}")
    with leg_metric_cols[4]:
        if np.isnan(leg2_vol_cents):
            st.metric("Leg 2 Daily P&L Vol (cents)", "n/a")
        else:
            st.metric("Leg 2 Daily P&L Vol (cents)", f"{abs(leg2_vol_cents):,.2f}")
    with leg_metric_cols[5]:
        if np.isnan(notional_leg2):
            st.metric("Leg 2 Notional Needed (EUR)", "n/a")
        else:
            st.metric(
                "Leg 2 Notional Needed (EUR)",
                format_compact(abs(notional_leg2), unit="€"),
            )

    carry_cols = st.columns(5, gap="large")
    st.markdown(
        f"Carry Roll Down Analysis over the following window (days): **{int(roll_vol)}**"
    )
    with carry_cols[0]:
        if np.isnan(leg1_carry_cents):
            st.metric("Leg 1 Carry+Roll (cents)", "n/a")
        else:
            st.metric("Leg 1 Carry+Roll (cents)", f"{leg1_carry_cents:,.2f}")
    with carry_cols[1]:
        if np.isnan(leg2_carry_cents):
            st.metric("Leg 2 Carry+Roll (cents)", "n/a")
        else:
            st.metric("Leg 2 Carry+Roll (cents)", f"{leg2_carry_cents:,.2f}")
    with carry_cols[2]:
        if np.isnan(trade_carry_cents):
            st.metric("Trade Carry+Roll (cents)", "n/a")
        else:
            st.metric("Trade Carry+Roll (cents)", f"{trade_carry_cents:,.2f}")
    with carry_cols[3]:
        if np.isnan(leg1_window_vol_cents):
            st.metric("Trade Window Vol (cents)", "n/a")
        else:
            st.metric("Trade Window Vol (cents)", f"{leg1_window_vol_cents:,.2f}")
    with carry_cols[4]:
        if np.isnan(carry_ratio_pct):
            st.metric("Trade Carry/Vol Ratio (%)", "n/a")
        else:
            st.metric("Trade Carry/Vol Ratio (%)", f"{carry_ratio_pct:,.2f}%")

    st.subheader("Carry-to-Vol Ratio vs Horizon")
    if np.isnan(trade_carry_cents) or np.isnan(ratio) or dv01_leg1 <= 0 or dv01_leg2 <= 0:
        st.info("Carry and volatility metrics required to compute horizon ratios.")
    else:
        horizons = [20, 60, 120, 252]
        ratio_points: list[dict[str, float]] = []
        series_for_vol = spread_returns_bps(spread) if spread is not None else pd.Series(dtype="float64")
        series_for_vol = series_for_vol.dropna()

        for horizon in horizons:
            if horizon <= 0 or series_for_vol.size < horizon:
                continue

            leg1_carry_h = leg_carry_roll(
                coupon_pct=coupon_leg1,
                repo_pct=repo_leg1,
                roll_down_bps=roll_down_leg1,
                gross_price=price_leg1,
                dv01_eur_bp=dv01_leg1,
                window_days=int(horizon),
            )
            leg2_carry_h = leg_carry_roll(
                coupon_pct=coupon_leg2,
                repo_pct=repo_leg2,
                roll_down_bps=roll_down_leg2,
                gross_price=price_leg2,
                dv01_eur_bp=dv01_leg2,
                window_days=int(horizon),
            )

            trade_carry_h = leg1_carry_h.total_cents - (leg2_carry_h.total_cents * ratio)

            horizon_vol_bp = series_for_vol.rolling(int(horizon)).std(ddof=1).iloc[-1]
            if pd.isna(horizon_vol_bp) or horizon_vol_bp <= 0:
                continue

            leg1_daily_vol_cents = horizon_vol_bp * dv01_leg1_cents
            leg1_window_vol_cents = leg1_daily_vol_cents * np.sqrt(float(horizon))

            if leg1_window_vol_cents > 0 and not np.isnan(trade_carry_h):
                ratio_points.append(
                    {
                        "horizon": horizon,
                        "ratio": (trade_carry_h / leg1_window_vol_cents) * 100.0,
                    }
                )

        ratio_df = pd.DataFrame(ratio_points).dropna(subset=["ratio"])

        if ratio_df.empty:
            st.info("No valid carry-to-vol ratios for the selected horizons.")
        else:
            ratio_chart = (
                alt.Chart(ratio_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("horizon:Q", title="Horizon (trading days)"),
                    y=alt.Y("ratio:Q", title="Carry / Expected Vol"),
                )
                .properties(height=260)
            )
            st.altair_chart(ratio_chart, use_container_width=True)

st.subheader("Bond Duration & DV01")
duration_rows = []

duration_keys = [
    (st.session_state.file1_name or "Leg 1", "Leg 1"),
    (st.session_state.file2_name or "Leg 2", "Leg 2"),
]

for display_name, key in duration_keys:
    info = duration_summary.get(key, {}) if duration_summary else {}
    duration_rows.append(
        {
            "Leg": display_name,
            "Yield (%)": info.get("yield_pct", float("nan")),
            "Macaulay Duration (yrs)": info.get("macaulay", float("nan")),
            "Modified Duration (yrs)": info.get("modified", float("nan")),
            "DV01 (cents)": info.get("dv01_cents", float("nan")),
        }
    )

duration_df = pd.DataFrame(duration_rows)
numeric_cols = [
    "Yield (%)",
    "Macaulay Duration (yrs)",
    "Modified Duration (yrs)",
    "DV01 (cents)",
]

if duration_df[numeric_cols].isna().all().all():
    st.info("Duration metrics unavailable for the current inputs.")
else:
    display_df = duration_df.copy()
    for col in numeric_cols:
        display_df[col] = display_df[col].apply(
            lambda x: "n/a" if pd.isna(x) else f"{x:,.4f}" if col == "Yield (%)" else f"{x:,.2f}"
        )
    st.table(display_df.set_index("Leg"))

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
        if "pnl" in locals() and pnl is not None:
            out["spread_return_bp"] = returns_bps
            out["pnl_eur"] = pnl
        st.download_button(
            "Download Spread Analytics (CSV)",
            out.to_csv().encode("utf-8"),
            file_name="spread_analytics.csv",
            mime="text/csv",
        )

# Footer notes
st.caption(
    "Tips: Use consistent date columns in your CSVs. You can toggle whether spread vol uses daily deltas "
    "or levels, and whether it is annualized. Z-Score is calculated on spread levels over a rolling window."
)
