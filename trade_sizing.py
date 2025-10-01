"""Helper functions for trade sizing calculations."""

from __future__ import annotations

import pandas as pd


def spread_returns_bps(spread: pd.Series) -> pd.Series:
    """Compute day-over-day spread changes in basis points.

    If spreads appear to be decimals (e.g., 0.052 for 5.2%) scale by 10,000.
    Otherwise assume they are in percent (e.g., 5.2) and scale by 100.
    """

    if spread is None or spread.empty:
        return pd.Series(dtype="float64")

    diffs = spread.diff()
    median_abs_diff = diffs.abs().median()

    if pd.isna(median_abs_diff):
        scale = 100.0
    else:
        # If daily moves are in percentage points (>= 0.0005), treat inputs as percent.
        scale = 100.0 if median_abs_diff >= 0.0005 else 10_000.0

    returns = diffs * scale
    returns.name = spread.name
    return returns


def pnl_series(returns_bps: pd.Series, dv01: float) -> pd.Series:
    """Convert spread returns (bp) into daily P&L using DV01."""

    if returns_bps is None or returns_bps.empty:
        return pd.Series(dtype="float64")

    pnl = returns_bps * float(dv01)
    pnl.name = "pnl"
    return pnl


def rolling_vol(series: pd.Series, window: int) -> float:
    """Return the latest rolling standard deviation for the supplied series."""

    if series is None or series.empty:
        return float("nan")

    vol = series.rolling(window).std(ddof=1).iloc[-1]
    return float(vol) if pd.notna(vol) else float("nan")
