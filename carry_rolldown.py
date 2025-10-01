"""Carry and roll-down helper calculations for bond legs and spreads."""

from __future__ import annotations

from dataclasses import dataclass


DAY_COUNT_DEFAULT = 365.0


def annual_fraction(window_days: int, day_count: float = DAY_COUNT_DEFAULT) -> float:
    """Return the year fraction represented by the rolling window length."""

    if window_days <= 0:
        return 0.0
    return float(window_days) / float(day_count)


@dataclass
class LegCarry:
    coupon_cents: float
    repo_cents: float
    roll_cents: float

    @property
    def carry_cents(self) -> float:
        return self.coupon_cents - self.repo_cents

    @property
    def total_cents(self) -> float:
        return self.carry_cents + self.roll_cents


def leg_carry_roll(
    *,
    coupon_pct: float,
    repo_pct: float,
    roll_down_bps: float,
    gross_price: float,
    dv01_eur_bp: float,
    window_days: int,
    day_count: float = DAY_COUNT_DEFAULT,
) -> LegCarry:
    """Compute coupon, repo, and roll-down carry over the window for a leg."""

    fraction = annual_fraction(window_days, day_count)

    coupon_cents = coupon_pct * 100.0 * fraction
    repo_cents = repo_pct * gross_price * fraction
    roll_cents = roll_down_bps * dv01_eur_bp * fraction

    return LegCarry(coupon_cents=coupon_cents, repo_cents=repo_cents, roll_cents=roll_cents)


def trade_carry_roll(
    leg1: LegCarry,
    leg2: LegCarry,
    ratio_notional: float,
) -> tuple[float, float, float]:
    """Return carry for leg1, leg2, and trade (EUR) using DV01 ratio."""

    if ratio_notional <= 0 or ratio_notional != ratio_notional:
        return float("nan"), float("nan"), float("nan")

    leg1_total = leg1.total_cents
    leg2_total = leg2.total_cents * ratio_notional
    trade_total = leg1_total - leg2_total
    return leg1_total, leg2_total, trade_total


def carry_vol_ratio(carry_cents: float, vol_eur: float) -> float:
    """Carry-to-vol ratio as a percent, converting carry cents to EUR."""

    if vol_eur <= 0 or vol_eur != vol_eur:
        return float("nan")
    carry_eur = carry_cents / 100.0
    return (carry_eur / vol_eur) * 100.0
