"""Generate the two images used in README.md.

Run from the repo root:
    uv run python scripts/_gen_readme_images.py

Outputs:
    docs/scenario_e_power_cap.png   — Scenario E LP schedule (power cap)
    docs/langgraph_structure.png    — LangGraph node structure
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

from aerogrid.config import (
    APPLIANCES, EV_DEADLINE_HOUR, HEATER_DEADLINES,
    HOUSE_POWER_CAP_KW, SLOT_MINUTES,
)
from aerogrid.optimizer import solve_receding_horizon

DOCS = REPO_ROOT / "docs"
DOCS.mkdir(exist_ok=True)

SLOT_H = SLOT_MINUTES / 60.0
_PALETTE = {
    "ev_charger": "#ff7f0e",
    "heater":     "#d62728",
}


# ── image 1: Scenario E ──────────────────────────────────────────────────────

def _utc(h: int, m: int = 0) -> datetime:
    return datetime(2026, 4, 15, h, m, tzinfo=timezone.utc)


def gen_scenario_e() -> Path:
    prices = np.full(96, 80.0)
    prices[4:8] = 20.0          # cheap slot at 22:00–23:00

    sched = solve_receding_horizon(
        _utc(21), prices,
        horizon_slots=96, remaining_ev_kwh=12.0, house_cap_kw=8.0,
    )

    T = sched.horizon_slots
    t0 = sched.slot_start
    times = [t0 + timedelta(minutes=SLOT_MINUTES * i) for i in range(T)]
    width = SLOT_MINUTES * 0.9 / (60.0 * 24.0)

    ev   = np.asarray(sched.ev_power_kw,    dtype=float)
    heat = np.asarray(sched.heater_power_kw, dtype=float)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 6),
        sharex=True, gridspec_kw={"height_ratios": [3, 2]},
    )
    fig.patch.set_facecolor("#0f1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="#c8cdd8")
        ax.xaxis.label.set_color("#c8cdd8")
        ax.yaxis.label.set_color("#c8cdd8")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2e3347")
        ax.grid(True, color="#2e3347", linewidth=0.6)

    ax1.bar(times, ev,   width=width, color="#ff7f0e", alpha=0.9, label="EV charger (kW)", align="edge")
    ax1.bar(times, heat, width=width, bottom=ev, color="#d62728", alpha=0.85, label="Heater (kW)", align="edge")
    ax1.axhline(8.0, color="#e8eaf6", ls="--", lw=1.6, label="Cap 8 kW")

    # EV deadline
    ev_dl = t0.replace(hour=EV_DEADLINE_HOUR, minute=0, second=0, microsecond=0)
    if ev_dl <= t0:
        ev_dl += timedelta(days=1)
    if ev_dl <= times[-1]:
        ax1.axvline(ev_dl, color="#ef5350", ls=":", lw=1.4, label="EV deadline 07:00")

    ax1.set_ylabel("load  (kW)", color="#c8cdd8")
    ax1.set_ylim(0, 11.5)
    ax1.legend(loc="upper right", framealpha=0.25, labelcolor="#e8eaf6",
               facecolor="#1a1d27", edgecolor="#2e3347", fontsize=9)

    title = (
        "Scenario E — 8 kW cap couples EV + heater\n"
        f"Cost €{sched.expected_cost:.3f}  ·  "
        f"Baseline €{sched.baseline_cost:.3f}  ·  "
        f"Savings {sched.savings()*100:.1f}%"
    )
    ax1.set_title(title, color="#e8eaf6", fontsize=11, pad=8)

    import matplotlib.dates as mdates
    ax2.step(times, prices, where="post", color="#29b6f6", lw=1.8, label="Price (€/MWh)")
    ax2.fill_between(times, prices, step="post", alpha=0.18, color="#29b6f6")
    ax2.set_ylabel("price  (€/MWh)", color="#c8cdd8")
    ax2.set_xlabel("time (UTC)", color="#c8cdd8")
    ax2.legend(loc="upper right", framealpha=0.25, labelcolor="#e8eaf6",
               facecolor="#1a1d27", edgecolor="#2e3347", fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=3))

    fig.tight_layout(pad=1.2)
    out = DOCS / "scenario_e_power_cap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {out}")
    return out


# ── image 2: LangGraph node diagram ─────────────────────────────────────────
# Layout: vertical pipeline (top → bottom) in the centre column.
# Environment boxes (Streamer, PriceServer) sit in a left column and feed
# into the pipeline via diagonal arrows.  Triggers fan-in above TriggerManager.
# No nested bounding boxes — sections are shown as tinted background rects
# drawn behind everything.

def gen_graph_diagram() -> Path:
    fig, ax = plt.subplots(figsize=(11, 11))
    BG = "#0d1117"
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(-0.7, 10.4)
    ax.axis("off")

    # ── geometry constants ───────────────────────────────────────────────────
    NW, NH = 2.8, 0.70          # main node width / height
    TW, TH = 1.15, 0.52         # trigger box width / height
    EW, EH = 2.1, 0.68          # environment box width / height
    R = 0.09                    # corner radius

    # x positions
    ENV_X  = 1.25               # environment column centre
    PIPE_X = 6.5                # pipeline column centre

    # y positions (pipeline, top → bottom)
    Y_TM   = 8.2                # TriggerManager
    Y_FC   = 6.7                # forecast_price
    Y_OPT  = 5.2                # optimize
    Y_PROP = 3.7                # propose_reschedule
    Y_HITL = 2.35               # hitl_gate
    Y_COM  = 1.05               # commit_plan
    Y_END  = 0.1                # END oval centre

    # trigger row
    Y_TRIG = 9.55
    TRIG_XS = [4.8, 6.0, 7.2, 8.4]

    # ── helpers ──────────────────────────────────────────────────────────────
    def _box(cx, cy, label, sub="", color="#1e88e5", w=NW, h=NH, edge="#90caf9"):
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle=f"round,pad={R}",
            facecolor=color, edgecolor=edge, linewidth=1.6, zorder=3,
        ))
        dy = 0.10 if sub else 0
        ax.text(cx, cy + dy, label, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="#ffffff", zorder=4)
        if sub:
            ax.text(cx, cy - 0.17, sub, ha="center", va="center",
                    fontsize=7.8, color="#b0bec5", zorder=4)

    def _arrow(x0, y0, x1, y1, label="", lx_off=0.0, ly_off=0.12,
               color="#78909c", rad=0.0):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5,
                                   connectionstyle=f"arc3,rad={rad}"),
                    zorder=2)
        if label:
            mx = (x0 + x1) / 2 + lx_off
            my = (y0 + y1) / 2 + ly_off
            ax.text(mx, my, label, ha="center", va="bottom",
                    fontsize=7.5, color="#90a4ae", zorder=5)

    def _oval(cx, cy, label):
        ax.add_patch(mpatches.Ellipse(
            (cx, cy), 1.3, 0.52,
            facecolor="#263238", edgecolor="#546e7a", lw=1.5, zorder=3,
        ))
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="#eceff1", zorder=4)

    # ── background sections (drawn first, behind everything) ─────────────────
    # Left: Digital Twin environment
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.12, 3.9), 2.27, 5.2,
        boxstyle="round,pad=0.1",
        facecolor="#0d2137", edgecolor="#1565c0", linewidth=1.2,
        linestyle="--", zorder=0,
    ))
    ax.text(1.25, 9.25, "Digital Twin", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="#42a5f5", zorder=1)
    ax.text(1.25, 8.95, "(shared environment)", ha="center", va="center",
            fontsize=7.5, color="#64b5f6", zorder=1)

    # Right: OptimizerStrategy
    ax.add_patch(mpatches.FancyBboxPatch(
        (3.55, -0.55), 6.7, 10.65,
        boxstyle="round,pad=0.1",
        facecolor="#0d1f0d", edgecolor="#2e7d32", linewidth=1.2,
        linestyle="--", zorder=0,
    ))
    ax.text(6.9, 10.18, "OptimizerStrategy  (autonomous agent)",
            ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="#66bb6a", zorder=1)

    # ── trigger boxes ────────────────────────────────────────────────────────
    trig_cfg = [
        ("new onset",      "#0d47a1"),
        ("price spike\n≥25%", "#4a148c"),
        ("EV deadline\nslip",  "#880e4f"),
        ("15-min\nresync",  "#1b5e20"),
    ]
    for tx, (lbl, col) in zip(TRIG_XS, trig_cfg):
        ax.add_patch(mpatches.FancyBboxPatch(
            (tx - TW/2, Y_TRIG - TH/2), TW, TH,
            boxstyle=f"round,pad=0.06",
            facecolor=col, edgecolor="#78909c", linewidth=1.0, zorder=3,
        ))
        ax.text(tx, Y_TRIG, lbl, ha="center", va="center",
                fontsize=8, color="#e3f2fd", zorder=4)

    # fan-in arrows: each trigger bottom → TriggerManager top
    for tx in TRIG_XS:
        _arrow(tx, Y_TRIG - TH/2, PIPE_X, Y_TM + NH/2,
               color="#37474f", rad=0.0)

    # ── pipeline nodes ───────────────────────────────────────────────────────
    pipeline = [
        (Y_TM,   "TriggerManager",     "30 s cooldown",           "#263238", "#90caf9"),
        (Y_FC,   "forecast_price",     "PriceOracle",             "#1565c0", "#90caf9"),
        (Y_OPT,  "optimize",           "LP / MIP  (HiGHS)",       "#4a148c", "#ce93d8"),
        (Y_PROP, "propose_reschedule", "RescheduleProposal?",      "#880e4f", "#f48fb1"),
        (Y_HITL, "hitl_gate",          "AUTO / ASK  (interrupt)", "#e65100", "#ffcc80"),
        (Y_COM,  "commit_plan",        "CommitTracker",            "#1b5e20", "#a5d6a7"),
    ]
    for y, lbl, sub, col, edge in pipeline:
        _box(PIPE_X, y, lbl, sub, color=col, edge=edge)

    # vertical arrows between consecutive pipeline nodes
    ys = [p[0] for p in pipeline]
    for ya, yb in zip(ys, ys[1:]):
        _arrow(PIPE_X, ya - NH/2, PIPE_X, yb + NH/2, color="#90caf9")

    # commit_plan → END
    _arrow(PIPE_X, Y_COM - NH/2, PIPE_X, Y_END + 0.26, color="#90caf9")
    _oval(PIPE_X, Y_END, "END")

    # ── environment nodes ────────────────────────────────────────────────────
    Y_STR = 7.55
    Y_PS  = 5.0
    _box(ENV_X, Y_STR, "Streamer",    "1 Hz ticks",    color="#1c313a", w=EW, h=EH,
         edge="#546e7a")
    _box(ENV_X, Y_PS,  "PriceServer", "SMARD DE-LU",   color="#1c313a", w=EW, h=EH,
         edge="#546e7a")

    # Streamer → TriggerManager
    _arrow(ENV_X + EW/2, Y_STR, PIPE_X - NW/2, Y_TM,
           label="samples + onsets", lx_off=0.5, ly_off=0.14,
           color="#546e7a", rad=-0.15)

    # PriceServer → forecast_price
    _arrow(ENV_X + EW/2, Y_PS, PIPE_X - NW/2, Y_FC,
           label="price history", lx_off=0.4, ly_off=0.14,
           color="#546e7a", rad=-0.1)

    out = DOCS / "langgraph_structure.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    gen_scenario_e()
    gen_graph_diagram()
    print("done")
