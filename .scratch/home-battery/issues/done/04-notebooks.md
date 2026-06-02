# Notebooks: battery LP scenarios + three-strategy comparison plots

Status: ready-for-agent

## Parent

`.scratch/home-battery/PRD.md` — Home Battery

## What to build

Refresh the demo notebooks so a reader can *see* the battery filling troughs,
emptying at peaks, and lowering the bill.

- **`05_optimizer`:** add Home Battery LP scenarios on cheap-overnight /
  expensive-day price curves, with plots showing the optimizer charging in
  cheap slots and discharging in expensive ones, and the SoC trajectory.
- **`06_end_to_end`:** drive the three-strategy digital-twin run and add:
  - a **state-of-charge trajectory vs price** plot (battery filling troughs,
    emptying at peaks),
  - a **charge/discharge-vs-price** plot (confirming buy-low / spend-high),
  - a **three-strategy cumulative-cost comparison** / cost waterfall quantifying
    the battery's savings,
  - a **representative-day dispatch** plot showing the battery covering the
    evening peak.

`03_price_oracle` is unaffected.

## Acceptance criteria

- [ ] `05_optimizer` demonstrates a battery LP scenario with charge/discharge
      and SoC plots against a price curve.
- [ ] `06_end_to_end` includes a SoC-vs-price trajectory plot.
- [ ] `06_end_to_end` includes a charge/discharge-vs-price plot.
- [ ] `06_end_to_end` includes a three-strategy cumulative-cost comparison
      quantifying the battery's savings.
- [ ] `06_end_to_end` includes a representative-day dispatch plot showing the
      battery covering the evening peak.
- [ ] Notebooks run top-to-bottom without error against the current code.

## Blocked by

- `.scratch/home-battery/issues/03-battery-live-loop.md` (plots consume the
  live three-strategy slot logs).
