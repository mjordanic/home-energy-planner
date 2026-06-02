# Bill-level Savings: net the discharge credit against a real Base-Load cost

Status: done

## Parent

`.scratch/battery-review-fixes/PRD.md` — Home Battery review fixes

## What to build

Measure forecast **Savings** at the whole-bill level so the reported number is
honest. Today `expected_cost` and `baseline_cost` both *exclude* the exogenous
**Base Load** cost, yet `expected_cost` *includes* the battery's discharge credit
(`−Σ p_dis·price`), which exists precisely to offset Base Load (ADR 0003). A slot
where the battery fully covers the evening Base-Load peak therefore reads as
*negative* forecast cost — the forecast-space twin of the illegal realized export
— and the battery's savings are systematically overstated.

End-to-end, this slice nets the discharge credit against a real Base-Load cost:

- `solve_receding_horizon` adds the Base-Load cost over the horizon
  (`Σ base_load_kw[t]·price[t]·per_slot_factor`) to `expected_cost`, and
  `_baseline_cost` adds the same term. A fully-shaved slot then reads ~0 (not
  negative); the headline ratio reflects money off the *total* bill — the only
  thing a Home Battery exists to reduce — and lines up with realized
  `cumulative_cost`, which already bills Base Load.
- When `base_load_kw` is `None`/zeros the added term is zero, so the no-battery /
  no-base-load output is byte-for-byte unchanged. Because the same non-negative
  term is added to both, `expected ≤ baseline` continues to hold.

Cleanups folded into this slice (same function, same `expected_cost` accounting):

- Compute `min_price` and `λ = (min_price/1000)·η_d` **once** in
  `solve_receding_horizon` and reuse it for both the objective's terminal-reward
  term and the `expected_cost` back-out, removing the duplicate computation that
  risks drift.
- Fix the stale Base-Load comment in `config.py` that claims two different
  daily-energy totals, leaving one consistent statement.

Expect the headline Savings percentage in the notebooks to *drop* — that is the
intended, honest outcome, not a regression.

## Acceptance criteria

- [ ] With a Base Load supplied, both `expected_cost` and `baseline_cost` increase
      by the Base-Load energy cost over the horizon.
- [ ] A battery slot that fully covers the Base-Load peak no longer yields a
      negative forecast cost (reads ~0).
- [ ] With `base_load_kw` absent (`None`/zeros), both `expected_cost` and
      `baseline_cost` are byte-for-byte unchanged (regression-clean), and all
      existing optimizer tests pass.
- [ ] `expected_cost ≤ baseline_cost` still holds.
- [ ] `min_price`/`λ` are computed in exactly one place and reused for both the
      objective and the `expected_cost` unpacking.
- [ ] The `config.py` Base-Load daily-energy comment states a single consistent
      total.

## Blocked by

None — can start immediately. (Independent code path from
`01-realized-no-export-throttle`; per the PRD, the two are the same coherence bug
in forecast vs realized space — review them together.)
