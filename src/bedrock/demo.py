"""Fundamentals (demo): SYNTHETIC sample data for FICTIONAL companies.

Shows how a fundamentals screen (earnings, dividends, balance sheet) would work once Rock Data Plus or company
filings provide real financial statements. Nothing here is real. The companies are invented so that no number
can be mistaken for a real company's, and no other module, chart or finding uses this data.

Scores are equal-weight percentile ranks, like the Opportunity Radar, and every company still has to pass the
same exit check ("can you get out?") before it counts as an opportunity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .radar import EXIT_DAYS

DEMO_SEED = 2026
LABEL = "SYNTHETIC DEMO DATA: fictional companies, not real figures"


def fundamentals_demo(n: int = 12, seed: int = DEMO_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "company": [f"Demo Company {c}" for c in "ABCDEFGHIJKLMNOP"[:n]],
        "sector": rng.choice(["Banking", "Telecom", "Manufacturing", "Agriculture", "Energy", "Insurance"], n),
        "pe_ratio": rng.uniform(3, 18, n).round(1),
        "dividend_yield": rng.uniform(0.0, 0.12, n).round(3),
        "return_on_equity": rng.uniform(-0.05, 0.30, n).round(3),
        "debt_to_equity": rng.uniform(0.1, 2.5, n).round(2),
        "earnings_growth": rng.normal(0.08, 0.15, n).round(3),
        "days_to_exit_10m": np.exp(rng.normal(2.6, 2.0, n)).round(1),
    })
    r = lambda s, asc=True: s.rank(pct=True, ascending=asc)  # noqa: E731
    d["value_score"] = (r(d["pe_ratio"], False) + r(d["dividend_yield"])) / 2
    d["quality_score"] = (r(d["return_on_equity"]) + r(d["debt_to_equity"], False) + r(d["earnings_growth"])) / 3
    d["score"] = (d["value_score"] + d["quality_score"]) / 2
    d["exit_ok"] = d["days_to_exit_10m"] <= EXIT_DAYS
    d["data"] = LABEL
    return d.sort_values("score", ascending=False).reset_index(drop=True)


def foreign_flows_demo(n_companies: int = 8, days: int = 60, seed: int = DEMO_SEED) -> pd.DataFrame:
    """SYNTHETIC daily foreign buying and selling for fictional companies, to show what a per-stock, daily sell-off
    alert would look like once Rock Data Plus provides foreign trading per stock. Not real; used nowhere else."""
    rng = np.random.default_rng(seed + 1)
    dates = pd.bdate_range(end="2026-11-16", periods=days)
    rows = []
    for i, c in enumerate("ABCDEFGHIJ"[:n_companies]):
        turnover = np.exp(rng.normal(16 - 0.4 * i, 0.6, days))           # KES traded a day
        drift = rng.normal(0, 0.08) - (0.25 if c in "BE" else 0)          # two companies under steady foreign selling
        net_share = np.clip(rng.normal(drift, 0.25, days), -0.9, 0.9)     # net foreign flow as a share of turnover
        for d, tv, s in zip(dates, turnover, net_share):
            rows.append({"date": d, "company": f"Demo Company {c}", "turnover_kes": round(tv, -3),
                         "net_foreign_kes": round(tv * s, -3), "net_foreign_share": round(s, 3), "data": LABEL})
    return pd.DataFrame(rows)


def foreign_alert_demo(flows: pd.DataFrame, window: int = 10, line: float = -0.15) -> pd.DataFrame:
    """Demo rule: flag a company when foreigners were net sellers of more than 15% of its trading over 10 days."""
    last = flows[flows["date"] >= flows["date"].drop_duplicates().nlargest(window).min()]
    g = last.groupby("company").agg(net_foreign_kes=("net_foreign_kes", "sum"), turnover_kes=("turnover_kes", "sum"))
    g["net_share"] = g["net_foreign_kes"] / g["turnover_kes"]
    g["signal"] = np.where(g["net_share"] <= line, "HIGH RISK (demo)", "NO CALL (demo)")
    return g.reset_index().sort_values("net_share")
