"""Bedrock dashboard.  Run:  streamlit run app/streamlit_app.py
Reads the outputs written by `python -m bedrock.pipeline` (or the committed app/snapshot/ when deployed).
"""
from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bedrock import stress  # noqa: E402
from bedrock.config import (DEFAULT_SCENARIO_DAYS, DEFAULT_SCENARIO_KES, HIGH_RISK_PROB,  # noqa: E402
                            PARTICIPATION, POSITION_KES, PROCESSED, STRATEGIC_CUTOFF)

OUT = PROCESSED / "outputs"
if not (OUT / "headline.json").exists():          # deployed (Streamlit Cloud): use the committed snapshot
    OUT = ROOT / "app" / "snapshot"
st.set_page_config(page_title="Bedrock · NSE fragility monitor", page_icon="🪨", layout="wide",
                   initial_sidebar_state="collapsed")

# ---------------------------------------------------------------- design tokens
INK, MUTED, FAINT, LINE, GRID, CARD = "#1B2420", "#5F6662", "#9AA09C", "#E5E1D8", "#EFECE5", "#FFFFFF"
ACCENT = "#B83A2B"                               # UI accent: risk badge, eyebrows
BLUE, RED = "#2a78d6", "#e34948"                 # chart series (validated reference palette)
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
TIER_COLORS = {"1 Deep": STATUS["good"], "2 Moderate": STATUS["warning"], "3 Thin": STATUS["serious"],
               "4 Frozen": STATUS["critical"]}
STRESS_COLORS = {"absorbed": STATUS["good"], "strained": STATUS["warning"], "trapped": STATUS["critical"]}
FONT = "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, .stApp, .stMarkdown, button, input {{ font-family: {FONT}; }}
header[data-testid="stHeader"] {{ display: none; }}
.block-container {{ padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1320px; }}
.bd-top {{ display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap; }}
.bd-brand {{ display: flex; align-items: center; gap: 10px; font-family: 'Space Grotesk', {FONT};
             font-weight: 700; font-size: 21px; color: {INK}; }}
.bd-mark {{ width: 24px; height: 24px; border-radius: 6px; background: {INK};
            box-shadow: inset 0 -7px 0 {ACCENT}; display: inline-block; }}
.bd-tagline {{ font-family: {FONT}; font-weight: 500; font-size: 14px; color: {MUTED}; margin-left: 4px; }}
.bd-asof {{ font-size: 13px; color: {MUTED}; }}
.bd-eyebrow {{ font-size: 12px; font-weight: 700; letter-spacing: .12em; color: {ACCENT}; margin: 30px 0 10px;
               display: flex; align-items: center; gap: 12px; }}
.bd-h1 {{ font-family: 'Space Grotesk', {FONT}; font-size: 34px; line-height: 1.14; font-weight: 600;
          color: {INK}; margin: 0 0 8px; max-width: 1180px; letter-spacing: -0.01em; }}
.bd-sub {{ font-size: 16px; line-height: 1.5; color: {MUTED}; margin: 0 0 22px; max-width: 820px; }}
.bd-badge {{ display: inline-block; background: {ACCENT}; color: #fff; font-weight: 700; font-size: 13px;
             letter-spacing: .08em; border-radius: 999px; padding: 5px 13px; vertical-align: middle; margin-right: 10px; }}
.bd-badge.quiet {{ background: {FAINT}; }}
.bd-kpis {{ display: grid; gap: 14px; margin: 0 0 14px; }}
.bd-kpi {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 16px; padding: 18px 20px 16px; }}
.bd-kpi .v {{ font-family: 'Space Grotesk', {FONT}; font-size: 34px; font-weight: 600; line-height: 1.05; color: {INK}; }}
.bd-kpi .v.risk {{ color: {ACCENT}; }}
.bd-kpi .l {{ font-size: 14px; line-height: 1.4; color: {MUTED}; margin-top: 8px; }}
.bd-model {{ display: flex; align-items: center; flex-wrap: wrap; gap: 6px 12px; font-size: 13.5px; color: {MUTED};
             border-top: 1px solid {LINE}; border-bottom: 1px solid {LINE}; padding: 10px 2px; margin: 4px 0 18px; }}
.bd-model .tag {{ font-size: 11px; font-weight: 700; letter-spacing: .1em; color: #fff; background: {INK};
                  border-radius: 5px; padding: 3px 8px; }}
.bd-model b {{ color: {INK}; font-weight: 600; }}
.bd-note {{ font-size: 14px; color: {MUTED}; margin: -4px 0 18px; }}
.bd-note b {{ color: {INK}; }}
.bd-cards {{ display: grid; gap: 14px; margin: 4px 0 18px; }}
.bd-card {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 18px; padding: 20px 22px; }}
.bd-card .k {{ font-size: 11.5px; font-weight: 700; letter-spacing: .12em; color: {MUTED}; }}
.bd-card .v {{ font-family: 'Space Grotesk', {FONT}; font-size: 34px; font-weight: 600; line-height: 1.05;
               color: {INK}; margin: 12px 0 8px; }}
.bd-card .v.risk {{ color: {ACCENT}; }}
.bd-card .d {{ font-size: 14.5px; line-height: 1.45; color: {MUTED}; }}
.bd-stock {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 14px; padding: 14px 18px; font-size: 15px;
             color: {MUTED}; line-height: 1.6; }}
.bd-stock b {{ color: {INK}; }}
.bd-foot {{ font-size: 12.5px; color: {FAINT}; margin-top: 30px; padding-top: 12px; border-top: 1px solid {LINE}; }}
[data-testid="stPlotlyChart"] {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 16px; padding: 0;
                                 overflow: hidden; }}
[data-testid="stExpander"] details {{ border: 1px solid {LINE}; border-radius: 12px; background: transparent; }}
[data-testid="stExpander"] summary p {{ font-size: 14.5px; font-weight: 500; color: {INK}; }}
@media (max-width: 1000px) {{
  .bd-cards {{ grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }}
  .bd-kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }}
  .bd-h1 {{ font-size: 28px; }}
}}
</style>
"""


# ---------------------------------------------------------------- helpers
def kes(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "-"
    for d, s in [(1e9, "bn"), (1e6, "m"), (1e3, "k")]:
        if abs(x) >= d:
            v = x / d
            return f"KES {v:,.1f}{s}".replace(".0" + s, s)
    return f"KES {x:,.0f}"


def html(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


def hero(eyebrow: str, title: str, sub: str = "", badge: str = "") -> None:
    b = f"<span class='bd-badge{'' if badge == 'HIGH RISK' else ' quiet'}'>{badge}</span>" if badge else ""
    html(f"<div class='bd-eyebrow'>{eyebrow}{b}</div><div class='bd-h1'>{title}</div>"
         + (f"<div class='bd-sub'>{sub}</div>" if sub else ""))


def kpis(items: list[tuple[str, str, bool]]) -> None:
    cells = "".join(f"<div class='bd-kpi'><div class='v{' risk' if r else ''}'>{v}</div><div class='l'>{l}</div></div>"
                    for v, l, r in items)
    html(f"<div class='bd-kpis' style='grid-template-columns:repeat({len(items)},minmax(0,1fr))'>{cells}</div>")


def model(name: str, *facts: str) -> None:
    html("<div class='bd-model'><span class='tag'>MODEL</span><b>" + name + "</b>"
         + "".join(f"<span>· {f}</span>" for f in facts) + "</div>")


def chart(fig: go.Figure, title: str = "", sub: str = "", height: int = 360, legend: bool = False,
          hbar: bool = False) -> None:
    t = (f"<b>{title}</b>" + (f"<br><span style='font-size:12.5px;color:{MUTED}'>{sub}</span>" if sub else "")) if title else None
    top = (16 + (40 if sub else 22) + (28 if legend else 0)) if title else 8
    fig.update_layout(
        height=height, template="none", paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(family=FONT, size=13, color=INK), showlegend=legend,
        title=dict(text=t, x=0, xref="container", xanchor="left", y=1, yref="container", yanchor="top",
                   pad=dict(t=16, l=20), font=dict(size=15, color=INK)) if t else None,
        margin=dict(l=20, r=22, t=top + 10, b=16),
        legend=dict(orientation="h", x=0, xanchor="left", y=1.0, yanchor="bottom", font=dict(size=12.5, color=MUTED),
                    title_text=""),
        hoverlabel=dict(bgcolor=INK, bordercolor=INK, font=dict(color="#fff", family=FONT, size=13)),
        shapedefaults=dict(line=dict(color=FAINT, width=1.5)), bargap=0.25)
    grid_x, grid_y = (True, False) if hbar else (False, True)
    fig.update_xaxes(showgrid=grid_x, gridcolor=GRID, zeroline=False, showline=not hbar, linecolor=LINE, ticks="",
                     tickfont=dict(color=MUTED), automargin=True, title=None)
    fig.update_yaxes(showgrid=grid_y, gridcolor=GRID, zeroline=False, showline=False, ticks="",
                     tickfont=dict(color=MUTED), automargin=True, title=None)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def keep(key: str, fallback):
    """on_change callback: if the active option is clicked again (value None), keep the previous choice."""
    def _cb():
        if st.session_state.get(key) is None:
            st.session_state[key] = st.session_state.get(f"_{key}_last", fallback)
        st.session_state[f"_{key}_last"] = st.session_state[key]
    return _cb


def short(name: str) -> str:
    import re
    return re.sub(r"\s+(Ltd|Plc|Limited|PLC|Holdings Ltd|Group Plc)\.?$", "", str(name)).strip()


def nice_date(s: str, fmt: str = "%d %b %Y") -> str:
    return pd.Timestamp(s).strftime(fmt).lstrip("0")


# ---------------------------------------------------------------- data
@st.cache_data
def load():
    need = OUT / "headline.json"
    if not need.exists():
        return None
    r = lambda n: pd.read_parquet(OUT / f"{n}.parquet")  # noqa: E731
    opt = lambda n: r(n) if (OUT / f"{n}.parquet").exists() else pd.DataFrame()  # noqa: E731
    return dict(
        head=json.loads(need.read_text()), live=json.loads((OUT / "selloff_live.json").read_text()),
        snap=r("depth_snapshot"), hist=r("depth_history"), flags=r("speculation_flags"),
        sb_data=r("selloff_dataset"), wf=r("selloff_walk_forward"),
        rb_depth=r("robustness_depth_sensitivity"), rb_sig=r("robustness_selloff_significance"),
        rb_cal=r("robustness_selloff_calibration"), rb_spec=r("robustness_speculation_bootstrap"),
        rb_sets=opt("robustness_selloff_feature_sets"), rb_block=opt("robustness_block_cap"),
        crash_sum=json.loads((OUT / "crash_summary.json").read_text()) if (OUT / "crash_summary.json").exists() else None,
        crash_live=opt("crash_live"), crash_dec=opt("crash_deciles"), crash_cmp=opt("crash_compare"),
        crash_year=opt("crash_by_year"), crash_rob=opt("crash_robust"), crash_imp=opt("crash_importance"),
    )


D = load()
if D is None:
    st.error("No outputs yet. Run `python -m bedrock.download` then `python -m bedrock.pipeline`.")
    st.stop()
H, LIVE, SNAP = D["head"], D["live"], D["snap"]
EQ = SNAP[SNAP["instrument"] == "Equity"].copy()
ASOF = H["data_as_of"]
BEST = LIVE["model"]
SIG = D["rb_sig"].set_index("model").loc[BEST]
WF = D["wf"]
FLAGS = D["flags"][D["flags"]["watch"]].copy()
PER = pd.Timestamp(LIVE["period"])
NXT = PER + (pd.offsets.MonthEnd(1) if LIVE.get("freq", "ME") == "ME" else pd.offsets.Week(1))
FLOWS = D["sb_data"][["period", "flow", "flow_3"]].dropna(subset=["flow"])


def stress_run(amount_bn=DEFAULT_SCENARIO_KES / 1e9, days=DEFAULT_SCENARIO_DAYS, part=PARTICIPATION,
               cut=STRATEGIC_CUTOFF):
    recent = FLAGS[FLAGS["date"] >= FLAGS["date"].max() - pd.Timedelta(days=60)]
    res = stress.run_scenario(SNAP, amount_bn * 1e9, days, part, cut, set(recent["code"]))
    return res, stress.summarize(res, amount_bn * 1e9)


def depth_history_fig() -> go.Figure:
    h = D["hist"].copy()
    h["as_of"] = pd.to_datetime(h["as_of"])
    f = go.Figure(go.Scatter(x=h["as_of"], y=h["pct_over_20d"], mode="lines", line=dict(color=BLUE, width=2.5),
                             fill="tozeroy", fillcolor="rgba(42,120,214,0.07)",
                             hovertemplate="%{x|%b %Y}: %{y:.0%}<extra></extra>"))
    for i, anchor, shift in [(0, "left", -18), (h["pct_over_20d"].idxmax(), "center", 16), (len(h) - 1, "right", -18)]:
        r = h.loc[i]
        f.add_annotation(x=r["as_of"], y=r["pct_over_20d"], text=f"<b>{r['pct_over_20d']:.0%}</b> {r['as_of']:%Y}",
                         showarrow=False, yshift=shift, xanchor=anchor, xshift=6 if anchor == "left" else 0,
                         font=dict(size=13, color=INK))
    f.update_yaxes(tickformat=".0%", range=[0, 1.05])
    return f


def flows_fig(since: str = "2019-01-01") -> go.Figure:
    fl = FLOWS[FLOWS["period"] >= since]
    f = go.Figure(go.Bar(x=fl["period"], y=fl["flow"] / 1e3, marker_color=np.where(fl["flow"] < 0, RED, BLUE),
                         marker_line_width=0, hovertemplate="%{x|%b %Y}: KES %{y:,.1f}bn<extra></extra>"))
    f.add_hline(y=LIVE["threshold_kes_m"] / 1e3, line_dash="dash", line_color=INK, line_width=1)
    f.update_yaxes(ticksuffix="bn")
    return f


# ---------------------------------------------------------------- page chrome
html(CSS)
html(f"<div class='bd-top'><div class='bd-brand'><span class='bd-mark'></span>Bedrock"
     f"<span class='bd-tagline'>NSE fragility monitor</span></div>"
     f"<div class='bd-asof'>Prices to {nice_date(ASOF['stocks'])} · foreign flows to "
     f"{nice_date(ASOF['flows'], '%b %Y')}</div></div>")
SCREENS = ["Overview", "Market depth", "Sell-off alert", "Crash risk", "Speculation", "Stress test"]
if D["crash_sum"] is None:
    SCREENS.remove("Crash risk")
screen = st.segmented_control("Screen", SCREENS, default="Overview", key="nav", label_visibility="collapsed",
                              on_change=keep("nav", "Overview")) or "Overview"

# ================================================================= OVERVIEW
if screen == "Overview":
    res0, sm = stress_run()
    n90 = int((FLAGS["date"] >= FLAGS["date"].max() - pd.Timedelta(days=90)).sum())
    eq_trapped = "EQTY" in set(res0.loc[res0["status"] == "trapped", "code"])
    hero("NSE FRAGILITY MONITOR", "The NSE is near record highs. Could you actually sell?",
         "Five checks on where Kenya’s stock market would break in the next sell-off.")
    sig = LIVE["signal"]
    cards = [
        ("MARKET DEPTH", f"{H['depth']['pct_over_20d']:.0%}", False,
         f"of shares take more than four weeks to sell {kes(POSITION_KES)}"),
        ("SELL-OFF ALERT", sig, sig == "HIGH RISK",
         f"{LIVE['probability']:.0%} chance of heavy foreign selling in {NXT:%B %Y}"),
        ("STRESS TEST", f"{sm['n_trapped']} stocks", False,
         f"trapped if foreigners sell {kes(DEFAULT_SCENARIO_KES)} in a week" + (", even Equity" if eq_trapped else "")),
        ("SPECULATION", f"{n90}", False, "unusual price jumps in the last 90 days, each one a lead to check"),
    ]
    CS = D["crash_sum"]
    if CS:
        top_name = D["crash_live"].iloc[0]["name"] if len(D["crash_live"]) else ""
        cards.insert(2, ("CRASH RISK", f"{CS['top10_rate'] / CS['base_rate']:.0f}×", False,
                         f"how often our riskiest 10% of stocks crash, vs the average. Riskiest now: "
                         f"{escape(short(top_name))}"))
    html(f"<div class='bd-cards' style='grid-template-columns:repeat({len(cards)},minmax(0,1fr))'>" + "".join(
        f"<div class='bd-card'><div class='k'>{k}</div><div class='v{' risk' if r else ''}'>{v}</div>"
        f"<div class='d'>{d}</div></div>" for k, v, r, d in cards) + "</div>")
    a, b = st.columns(2, gap="medium")
    with a:
        chart(depth_history_fig(), "The rally lifted prices, not depth",
              "Share of shares needing more than 20 trading days to sell KES 10m", 340)
    with b:
        fl = FLOWS[FLOWS["period"] >= "2019-01-01"]
        chart(flows_fig(), f"Foreigners have pulled out KES {-fl['flow'].sum() / 1e3:,.0f}bn since 2019",
              "Net foreign flow into NSE shares, KES bn a month · dashed line = heavy selling", 340)

# ================================================================= MARKET DEPTH
elif screen == "Market depth":
    pos = st.session_state.get("pos") or POSITION_KES
    eq = EQ.copy()
    eq["days"] = (pos / (PARTICIPATION * eq["adv_kes"].replace(0, np.nan))).fillna(np.inf)
    pct20 = (eq["days"] > 20).mean()
    hero("MARKET DEPTH", f"{pct20:.0%} of NSE shares take more than four weeks to sell {kes(pos)}",
         f"{len(eq)} ordinary shares, selling no more than {PARTICIPATION:.0%} of a normal day’s trading.")
    st.segmented_control("Position to sell", [1e6, 10e6, 50e6, 100e6], default=POSITION_KES, format_func=kes,
                         key="pos", on_change=keep("pos", POSITION_KES))
    kpis([(f"{pct20:.0%}", "need more than 20 trading days", True),
          (f"{int((eq['days'] <= 2).sum())}", "stocks can be sold within 2 days", False),
          (f"{H['depth']['top5_share_of_value']:.0%}", "of all trading is in just 5 stocks", False),
          (f"{np.isinf(eq['days']).mean():.0%}", "of shares had no trades in 6 months", False)])
    model("Days-to-exit + K-means tiers", "120 days of trading per stock",
          "result holds at 62–72% under every assumption tested")

    a, b = st.columns([1, 1], gap="medium")
    with a:
        bands = [("≤ 2 days", 2, STATUS["good"]), ("2–5", 5, STATUS["warning"]), ("5–20", 20, STATUS["warning"]),
                 ("20–250", 250, STATUS["serious"]), ("> 250 or none", np.inf, STATUS["critical"])]
        cut = pd.cut(eq["days"], [-np.inf] + [x[1] for x in bands], labels=[x[0] for x in bands])
        n = cut.value_counts().reindex([x[0] for x in bands]).fillna(0).astype(int)
        f = go.Figure(go.Bar(x=n.index, y=n.values, marker_color=[x[2] for x in bands], marker_line_width=0,
                             text=[f"{v}" for v in n.values], textposition="outside", cliponaxis=False,
                             textfont=dict(color=INK, size=14),
                             hovertemplate="%{x} days: %{y} stocks<extra></extra>"))
        f.update_yaxes(showticklabels=False, showgrid=False, range=[0, n.max() * 1.18])
        chart(f, "Most shares take weeks or months to sell", f"Number of stocks by trading days to sell {kes(pos)}", 360)
    with b:
        chart(depth_history_fig(), "The market thinned for a decade",
              "Share needing more than 20 days to sell KES 10m, each quarter", 360)

    names = eq.sort_values("name")
    c1, c2 = st.columns([1, 2], gap="medium")
    pick = c1.selectbox("Look up a stock", names["code"], format_func=lambda c: f"{names.set_index('code').loc[c, 'name']} ({c})",
                        index=int(np.argmax(names["code"].values == "SCOM")) if "SCOM" in names["code"].values else 0)
    r = eq.set_index("code").loc[pick]
    d_txt = "no trades in 6 months" if np.isinf(r["days"]) else (
        "under a day" if r["days"] < 1 else f"{r['days']:,.0f} trading days")
    c2.markdown(f"<div class='bd-stock'><b>{escape(r['name'])}</b> · {escape(str(r['sector']))}<br>"
                f"Selling {kes(pos)} takes <b>{d_txt}</b> · trades about <b>{kes(r['adv_kes'])}</b> a day · "
                f"<b>{'n/a' if pd.isna(r['foreign_pct']) else format(r['foreign_pct'], '.0%')}</b> foreign-owned · tier <b>{r['tier'][2:]}</b></div>",
                unsafe_allow_html=True)

    with st.expander(f"All {len(eq)} stocks"):
        t = eq.sort_values("days")[["name", "sector", "tier", "days", "adv_kes", "foreign_pct"]].copy()
        t["tier"] = t["tier"].str[2:]
        st.dataframe(t, hide_index=True, width="stretch", column_config={
            "name": "Company", "sector": "Sector", "tier": "Tier",
            "days": st.column_config.NumberColumn("Days to sell", format="%.1f"),
            "adv_kes": st.column_config.NumberColumn("Trades a day (KES)", format="%,.0f"),
            "foreign_pct": st.column_config.NumberColumn("Foreign-owned", format="percent")})
    with st.expander("How we measure this, and does it hold?"):
        st.markdown("**Days to sell** = position ÷ (20% × the stock’s average daily value traded, last 120 days). "
                    "Tiers: Deep ≤ 2 days · Moderate ≤ 20 · Thin ≤ 250 · Frozen > 250 or no trades. "
                    "One-off block deals (e.g. NCBA’s KES 100bn transfer on 3 Nov 2026) count for at most 10× a "
                    "normal day.")
        rd = D["rb_depth"]
        piv = rd.pivot_table(index=["lookback_days", "participation"], columns="position_kes", values="pct_over_20d")
        piv.columns = [kes(c) for c in piv.columns]
        piv.index = [f"{lb}-day window, sell ≤ {p:.0%} a day" for lb, p in piv.index]
        st.dataframe(piv.style.format("{:.0%}"), width="stretch")
        if len(D["rb_block"]):
            bc = D["rb_block"][["block_cap", "pct_over_20d", "top5_share_of_value"]]
            st.dataframe(bc, hide_index=True, width="stretch", column_config={
                "block_cap": "Block deals counted at most",
                "pct_over_20d": st.column_config.NumberColumn("Need > 20 days", format="percent"),
                "top5_share_of_value": st.column_config.NumberColumn("Top-5 share", format="percent")})

# ================================================================= SELL-OFF ALERT
elif screen == "Sell-off alert":
    p, sig = LIVE["probability"], LIVE["signal"]
    last = FLOWS.set_index("period").loc[PER]
    title = (f"High risk of heavy foreign selling in {NXT:%B %Y}" if sig == "HIGH RISK"
             else f"No warning for {NXT:%B %Y}")
    hero("SELL-OFF ALERT", title,
         f"The model puts it at <b>{p:.0%}</b>, against {LIVE['base_rate']:.0%} in a normal month. Trigger: "
         f"{kes(-last['flow'] * 1e6)} net foreign outflow in {PER:%B} ({kes(-last['flow_3'] * 1e6)} over 3 months)."
         if last["flow"] < 0 else f"The model puts it at <b>{p:.0%}</b>, against {LIVE['base_rate']:.0%} normally.",
         badge=sig)
    hits, calls = int(SIG["hits"]), int(SIG["high_risk_calls"])
    a_ = WF["actual"].astype(bool)
    kpis([(f"{SIG['hit_rate']:.0%}", f"of its warnings were followed by heavy selling ({SIG['base_rate']:.0%} by chance)", True),
          (f"{hits} of {int(a_.sum())}", "heavy-selling months it warned about in advance", False),
          ("< 1%" if SIG["p_value_vs_base"] < 0.01 else f"{SIG['p_value_vs_base']:.0%}", "chance this record is luck", False)])
    rec = WF.tail(15)
    caught = int((rec["actual"].astype(bool) & (rec[BEST] >= HIGH_RISK_PROB)).sum())
    html(f"<div class='bd-note'><b>Weak spot:</b> it misses about half of all sell-offs, and caught {caught} of the last "
         f"{int(rec['actual'].sum())}. Weekly data from Rock is the fix.</div>")
    model("XGBoost classifier", "inputs: net foreign flow this month and over 3 months",
          f"trained on {LIVE['n_train']} months", f"tested on {len(WF)} months it had never seen")

    a, b = st.columns([1, 1], gap="medium")
    with a:
        chart(flows_fig(), "Foreign investors keep selling",
              "Net foreign flow into NSE shares, KES bn a month · dashed line = heavy selling", 380)
    with b:
        f = go.Figure()
        f.add_scatter(x=WF["period"], y=WF[BEST], mode="lines", name="predicted chance",
                      line=dict(color=BLUE, width=2), hovertemplate="%{x|%b %Y}: %{y:.0%}<extra></extra>")
        hit = WF[WF["actual"] == 1]
        f.add_scatter(x=hit["period"], y=hit[BEST], mode="markers", name="heavy selling followed",
                      marker=dict(color=RED, size=9, line=dict(color="white", width=2)),
                      hovertemplate="%{x|%b %Y}: heavy selling next month<extra></extra>")
        f.add_hline(y=HIGH_RISK_PROB, line_dash="dash", line_color=INK, line_width=1,
                    annotation_text="warning line (45%)", annotation_position="top right",
                    annotation_font=dict(size=12, color=MUTED))
        f.update_yaxes(tickformat=".0%", range=[0, 1])
        chart(f, "Its warnings against what happened", "Each month predicted using only earlier data", 380, legend=True)

    with st.expander("Full test results"):
        rows = []
        for thr in [0.35, 0.40, 0.45, 0.50]:
            m = WF[BEST] >= thr
            k, n = int(a_[m].sum()), int(m.sum())
            rows.append({"Warning line": f"{thr:.0%}", "Warnings": n, "Right": k, "Hit rate": k / n if n else np.nan,
                         "p-value": binomtest(k, n, a_.mean(), alternative="greater").pvalue if n else np.nan})
        st.markdown("**Does it depend on where the warning line is drawn?** No:")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            "Hit rate": st.column_config.NumberColumn(format="percent"),
            "p-value": st.column_config.NumberColumn(format="%.3f")})
        if len(D["rb_sets"]):
            st.markdown("**All six versions we tested** (we report every one, not just the best):")
            rs = D["rb_sets"][["inputs", "model", "high_risk_calls", "hits", "hit_rate", "p_value_vs_base",
                               "p_adjusted_bonferroni"]]
            st.dataframe(rs, hide_index=True, width="stretch", column_config={
                "inputs": "Inputs", "model": "Model", "high_risk_calls": "Warnings", "hits": "Right",
                "hit_rate": st.column_config.NumberColumn("Hit rate", format="percent"),
                "p_value_vs_base": st.column_config.NumberColumn("p-value", format="%.3f"),
                "p_adjusted_bonferroni": st.column_config.NumberColumn("p, adjusted for 6 tries", format="%.3f")})
        st.markdown("**When it says X%, how often did it happen?**")
        st.dataframe(D["rb_cal"], hide_index=True, width="stretch", column_config={
            "bin": "Predicted", "n": "Months",
            "predicted": st.column_config.NumberColumn("Avg predicted", format="percent"),
            "actual": st.column_config.NumberColumn("Happened", format="percent")})
        st.caption("Low readings were followed by heavy selling almost as often as normal, so the alert never says "
                   "“low risk”: only HIGH RISK or NO CALL. With about 140 monthly data points, adding economic inputs "
                   "made it worse, so the live model uses foreign flows only.")

# ================================================================= CRASH RISK
elif screen == "Crash risk":
    CS, CL, CD = D["crash_sum"], D["crash_live"], D["crash_dec"]
    drop = abs(CS["crash_drop"])
    hero("CRASH RISK", "The stocks Bedrock ranks riskiest crash twice as often as average",
         f"Each month every stock is ranked by its chance of falling {drop:.0%} or more over the next "
         f"{CS['horizon_days']} trading days. Tested on {CS['n_test']:,} stock-months it had never seen "
         f"({CS['first_test_year']}–{pd.Timestamp(CS['as_of']).year}).")
    kpis([(f"{CS['top10_rate']:.1%}", "of the riskiest 10% crashed the next month", True),
          (f"{CS['base_rate']:.1%}", "average crash rate across all stocks", False),
          (f"{CS['safest10_rate']:.1%}", "of the safest 10% crashed", False),
          (f"{CS['rule_top10_rate']:.1%}", "for a simple rule: the most volatile 10%", False)])
    html("<div class='bd-note'><b>Honest result:</b> crash risk is predictable, but mostly from volatility, so a "
         "simple rule does almost as well. Richer data from Rock (foreign trading per stock, order books) is "
         "where machine learning can add more.</div>")
    model("XGBoost classifier", "14 inputs: momentum, volatility, liquidity, spikes, market, foreign flows",
          f"trained year by year, tested on {CS['n_test']:,} stock-months")

    a, b = st.columns([1, 1], gap="medium")
    with a:
        top = CL.head(10).iloc[::-1].copy()
        top["label"] = top["name"].map(short)
        f = go.Figure(go.Bar(
            x=top["crash_prob"], y=top["label"], orientation="h", marker_color=RED, marker_line_width=0,
            customdata=np.c_[top["mom_3m"], top["vol_60"] * np.sqrt(250), top["group_crash_rate"]],
            hovertemplate="<b>%{y}</b><br>risk score %{x:.0%}<br>last 3 months %{customdata[0]:+.0%}"
                          "<br>volatility (yearly) %{customdata[1]:.0%}<br>its group crashed %{customdata[2]:.0%} "
                          "of the time<extra></extra>"))
        f.update_xaxes(tickformat=".0%")
        chart(f, f"Riskiest stocks on {nice_date(CS['as_of'])}", "Model risk score · hover for the evidence",
              400, hbar=True)
    with b:
        d = CD.copy()
        shades = [f"rgba(184,58,43,{0.18 + 0.82 * i / 9:.2f})" for i in range(10)]
        f = go.Figure(go.Bar(x=d["decile"].astype(str), y=d["crash_rate"], marker_color=shades, marker_line_width=0,
                             text=[f"{v:.0%}" for v in d["crash_rate"]], textposition="outside", cliponaxis=False,
                             textfont=dict(color=INK, size=12.5),
                             hovertemplate="risk group %{x}: %{y:.1%} crashed<extra></extra>"))
        f.add_hline(y=CS["base_rate"], line_dash="dash", line_color=INK, line_width=1,
                    annotation_text="average", annotation_position="top left",
                    annotation_font=dict(size=12, color=MUTED))
        f.update_yaxes(tickformat=".0%", range=[0, d["crash_rate"].max() * 1.25])
        f.update_xaxes(title=None, tickvals=list(d["decile"].astype(str)),
                       ticktext=["1<br>safest"] + [str(i) for i in range(2, 10)] + ["10<br>riskiest"])
        chart(f, "The higher the risk group, the more often stocks crashed",
              f"Share that fell {drop:.0%}+ in the next month, by monthly risk group (test years only)", 400)

    with st.expander(f"All {len(CL)} stocks, ranked"):
        tb = CL[["risk_rank", "name", "crash_prob", "group_crash_rate", "mom_3m", "dd_1y"]].copy()
        tb["name"] = tb["name"].map(short)
        st.dataframe(tb, hide_index=True, width="stretch", column_config={
            "risk_rank": "Rank", "name": "Company",
            "crash_prob": st.column_config.NumberColumn("Risk score", format="percent"),
            "group_crash_rate": st.column_config.NumberColumn("Its group crashed", format="percent"),
            "mom_3m": st.column_config.NumberColumn("Last 3 months", format="percent"),
            "dd_1y": st.column_config.NumberColumn("Below 1-year high", format="percent")})
    with st.expander("Full test results"):
        st.markdown("**Model against simple alternatives** (same test months):")
        st.dataframe(D["crash_cmp"][["model", "auc", "top10_crash_rate", "lift", "caught_in_top20", "p_value_top10"]],
                     hide_index=True, width="stretch", column_config={
                         "model": "Method", "auc": st.column_config.NumberColumn("AUC (ranking)", format="%.3f"),
                         "top10_crash_rate": st.column_config.NumberColumn("Riskiest 10% crashed", format="percent"),
                         "lift": st.column_config.NumberColumn("× average", format="%.1f"),
                         "caught_in_top20": st.column_config.NumberColumn("Crashes caught in riskiest 20%",
                                                                          format="percent"),
                         "p_value_top10": st.column_config.NumberColumn("p-value", format="%.4f")})
        g = CS["auc_gap_vs_rule"]
        st.caption(f"Ranking power (AUC) of the model minus the volatility rule: {g['gap']:+.3f}, 95% range "
                   f"{g['ci_low']:+.3f} to {g['ci_high']:+.3f}. That range includes zero: the model is not clearly "
                   "better than the rule at ranking, though its riskiest 10% crashed somewhat more often.")
        st.markdown("**Year by year:**")
        st.dataframe(D["crash_year"], hide_index=True, width="stretch", column_config={
            "year": st.column_config.NumberColumn("Year", format="%d"), "stock_months": "Stock-months",
            "crashes": "Crashes", "base_rate": st.column_config.NumberColumn("Average", format="percent"),
            "auc_model": st.column_config.NumberColumn("AUC model", format="%.2f"),
            "auc_volatility": st.column_config.NumberColumn("AUC rule", format="%.2f"),
            "top10_crash_rate": st.column_config.NumberColumn("Riskiest 10% crashed", format="percent")})
        st.markdown("**Other crash definitions:**")
        st.dataframe(D["crash_rob"], hide_index=True, width="stretch", column_config={
            "crash_definition": "Crash means", "base_rate": st.column_config.NumberColumn("Average", format="percent"),
            "auc_model": st.column_config.NumberColumn("AUC model", format="%.3f"),
            "auc_volatility": st.column_config.NumberColumn("AUC rule", format="%.3f")})
        st.caption("Raw model scores run higher than the crash rates that followed, so the dashboard reports each "
                   "stock’s risk group and that group’s real crash rate, not the score as a probability.")

# ================================================================= SPECULATION
elif screen == "Speculation":
    days = st.session_state.get("win") or 90
    last = FLAGS["date"].max()
    recent = FLAGS[FLAGS["date"] >= last - pd.Timedelta(days=days)].sort_values("date", ascending=False)
    news = FLAGS["news"].fillna("")
    n_yes, n_no = int(news.str.startswith("yes").sum()), int((news == "no news found").sum())
    bs = D["rb_spec"].set_index(["size", "horizon"])
    hero("SPECULATION", f"{len(recent)} unusual price jumps in the last {days} days",
         "Price and trading both broke sharply from the stock’s own normal while the market stayed calm. "
         "Each one is a lead to check, not proof.")
    st.segmented_control("Window", [30, 90, 180, 365], default=90, format_func=lambda d: f"Last {d} days", key="win",
                         on_change=keep("win", 90))
    kpis([(f"{len(recent)}", f"flags across {recent['code'].nunique()} stocks", False),
          (f"{n_no} of {n_yes + n_no}", "flags we checked had no news behind them", True),
          (f"{bs.loc[('small', 'fwd_5d'), 'gap']:+.1%}", "extra gain for small caps in the week after a flag", False),
          (f"{bs.loc[('large', 'fwd_20d'), 'gap']:+.1%}", "large caps give back over the next month", False)])
    model("Isolation Forest anomaly detector", "each stock against its own last 60 days",
          "market-wide moves removed", "flags checked by hand against news")

    a, b = st.columns([1.35, 1], gap="medium")
    with a:
        t = recent[["date", "name", "ret", "value_kes", "news"]].copy()
        t["news"] = t["news"].fillna("").replace("", "not checked yet")
        t["ret"] = t["ret"] * 100
        st.dataframe(t, hide_index=True, width="stretch", height=420, column_config={
            "date": st.column_config.DateColumn("Date", format="D MMM YYYY"), "name": "Company",
            "ret": st.column_config.NumberColumn("Day’s move", format="%+.1f%%"),
            "value_kes": st.column_config.NumberColumn("Traded (KES)", format="%,.0f"),
            "news": "News found?"})
    with b:
        g = D["rb_spec"].copy()
        g["label"] = g["size"].str.capitalize() + " caps, " + g["horizon"].str.extract(r"(\d+)")[0] + " days"
        g = g.iloc[::-1]
        col = np.where(~g["ci_excludes_zero"], FAINT, np.where(g["gap"] > 0, BLUE, RED))
        f = go.Figure(go.Scatter(
            x=g["gap"], y=g["label"], mode="markers",
            marker=dict(color=col, size=13, line=dict(color="white", width=2)),
            error_x=dict(type="data", symmetric=False, array=g["ci_high"] - g["gap"], arrayminus=g["gap"] - g["ci_low"],
                         color=FAINT, thickness=2, width=5),
            hovertemplate="%{y}: %{x:+.1%}<extra></extra>"))
        f.add_vline(x=0, line_color=INK, line_width=1)
        f.update_xaxes(tickformat="+.0%")
        chart(f, "After a flag, small caps keep running",
              "Extra return vs other big up-days · grey = could be chance", 420, hbar=True)

    with st.expander("How it works"):
        st.markdown("Every trading day since 2013, each stock’s return and value traded are compared with its own "
                    "previous 60 days, after removing the market’s move. An Isolation Forest scores how unusual each "
                    "day is; the most extreme go on the watch list. We then check the top flags against NSE "
                    "announcements and the press. Example: TransCentury rose 230% in 13 days in early 2025 with no "
                    "news; Bedrock flagged it on the second day.")

# ================================================================= STRESS TEST
else:
    ss = st.session_state
    amount, days = ss.get("amt", DEFAULT_SCENARIO_KES / 1e9), ss.get("days", DEFAULT_SCENARIO_DAYS)
    part, cut = ss.get("part", PARTICIPATION), ss.get("cut", STRATEGIC_CUTOFF)
    res, sm = stress_run(amount, days, part, cut)
    trapped = set(res.loc[res["status"] == "trapped", "code"])
    hero("STRESS TEST", f"Selling {kes(amount * 1e9)} in {days} days would trap {sm['n_trapped']} stocks"
         + (", even Equity" if "EQTY" in trapped else ""),
         "Trapped = needs more than four times the selling window to clear, or has no buyers at all.")
    c1, c2, c3 = st.columns([1, 1, 0.8], gap="large")
    c1.slider("Foreign selling (KES bn)", 0.5, 10.0, DEFAULT_SCENARIO_KES / 1e9, 0.5, key="amt")
    c2.slider("Over how many trading days", 1, 20, DEFAULT_SCENARIO_DAYS, key="days")
    with c3.expander("Advanced assumptions"):
        st.slider("Most you can sell in a day (share of normal trading)", 0.05, 0.5, PARTICIPATION, 0.05,
                  format="%.2f", key="part")
        st.slider("Foreign stakes above this are strategic and not sold", 0.3, 1.0, STRATEGIC_CUTOFF, 0.05,
                  format="%.2f", key="cut")
    kpis([(f"{sm['n_hit']}", "stocks hit by the selling", False),
          (f"{sm['n_trapped']}", "trapped: can’t be sold in time", True),
          (kes(sm["value_trapped_kes"]), "of selling stuck in trapped stocks", False),
          (f"{sm['weighted_impact']:.1%}", "average price fall, weighted by size: the index looks calm", False)])
    model("Square-root market-impact model", "selling split by foreign ownership (CMA, Aug 2026)",
          f"stakes above {cut:.0%} treated as strategic")

    top = res[(res["sell_kes"] > 0) & res["price_impact"].notna()].nsmallest(15, "price_impact").iloc[::-1]
    f = go.Figure()
    for s_, col in STRESS_COLORS.items():
        t = top[top["status"] == s_]
        if len(t):
            f.add_bar(x=t["price_impact"], y=t["name"].str.replace(r"\s+(Ltd|Plc|Limited|PLC)\.?$", "", regex=True),
                      orientation="h", name=s_, marker_color=col, marker_line_width=0,
                      customdata=np.c_[t["sell_kes"], t["days_to_absorb"]],
                      hovertemplate="<b>%{y}</b><br>price impact %{x:.1%}<br>sold KES %{customdata[0]:,.0f}"
                                    "<br>days to clear %{customdata[1]:,.0f}<extra></extra>")
    names = top["name"].str.replace(r"\s+(Ltd|Plc|Limited|PLC)\.?$", "", regex=True)
    f.update_layout(barmode="overlay", yaxis=dict(categoryorder="array", categoryarray=list(names)))
    f.update_xaxes(tickformat=".0%")
    chart(f, "The 15 hardest-hit stocks", "Estimated price fall from the selling", 540, legend=True, hbar=True)

    with st.expander("All stocks in this scenario"):
        st.dataframe(res[["name", "status", "foreign_pct", "sell_kes", "days_to_absorb", "price_impact"]],
                     hide_index=True, width="stretch", column_config={
                         "name": "Company", "status": "Status",
                         "foreign_pct": st.column_config.NumberColumn("Foreign-owned", format="percent"),
                         "sell_kes": st.column_config.NumberColumn("Sold (KES)", format="%,.0f"),
                         "days_to_absorb": st.column_config.NumberColumn("Days to clear", format="%.1f"),
                         "price_impact": st.column_config.NumberColumn("Price impact", format="percent")})
    with st.expander("How it works"):
        st.markdown("The selling is split across stocks by how much foreigners own (CMA ownership data). Each stock "
                    "can absorb at most the chosen share of its normal daily trading. Price impact uses the standard "
                    "square-root law: volatility × √(amount sold ÷ normal daily trading). Absorbed = clears within the "
                    "window · strained = within 4× · trapped = longer, or no trades in 120 days.")

html("<div class='bd-foot'>Sources: NSE daily prices (Mendeley Data, Wanjawa, CC BY 4.0; extended to "
     f"{nice_date(ASOF['stocks'])} from NSE daily price lists) · CMA Quarterly Statistical Bulletins · Central Bank of "
     "Kenya · FRED. Built to run on Rock Data Plus.</div>")
