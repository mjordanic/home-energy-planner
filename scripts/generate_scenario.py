"""Generate a scenario (mains + per-appliance + onsets parquet + MANIFEST).

Default window matches ``SCENARIO_TRAIN_START``/``SCENARIO_TEST_END`` so the
behavioral predictor's train/test split and the digital-twin simulation line
up out of the box.

Usage:
    python scripts/generate_scenario.py                 # full 90-day default
    python scripts/generate_scenario.py --days 3        # fast dev run
    python scripts/generate_scenario.py --seed 7
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aerogrid.config import (
    SCENARIO_DIR,
    SCENARIO_TEST_END,
    SCENARIO_TEST_START,
    SCENARIO_TRAIN_START,
)
from aerogrid.sim.scenario import (
    ScenarioGenerator,
    default_scenario_spec,
    write_scenario_parquet,
)
from aerogrid.logging_config import setup_logging
from scripts._common import write_manifest

logger = logging.getLogger(__name__)


def main() -> int:
    """Generate a deterministic household scenario and write it to parquets.

    Builds a :class:`~aerogrid.sim.scenario.ScenarioSpec` with the default
    appliance configuration (dishwasher, washing machine, EV charger), expands
    it to 1 Hz traces via :class:`~aerogrid.sim.scenario.ScenarioGenerator`,
    writes mains + per-appliance + onsets parquets, and stamps a
    ``MANIFEST.json`` with metadata.

    Returns:
        0 on success.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=SCENARIO_DIR)
    ap.add_argument("--days", type=int, default=None,
                    help="cap the scenario length (default: full train+test window).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level (default: INFO).",
    )
    args = ap.parse_args()

    setup_logging(level=getattr(logging, args.log_level, logging.INFO), console=True)

    start = SCENARIO_TRAIN_START
    end = SCENARIO_TEST_END
    if args.days is not None:
        end = start + timedelta(days=args.days)

    logger.info(
        "generate_scenario: start=%s end=%s days=%d seed=%d",
        start.isoformat(), end.isoformat(), (end - start).days, args.seed,
    )
    spec = default_scenario_spec(start, end, seed=args.seed)
    out = ScenarioGenerator().generate(spec)
    logger.info(
        "generate_scenario: generated mains_rows=%d onsets=%d appliances=%s",
        len(out.mains), len(out.onsets), list(out.per_appliance.keys()),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = write_scenario_parquet(out, args.out_dir)
    for label, p in paths.items():
        size_mb = p.stat().st_size / 1e6
        logger.info("generate_scenario: wrote %s → %s (%.1f MB)", label, p, size_mb)

    write_manifest(
        args.out_dir / "MANIFEST.json",
        source="simulated",
        url_base=None,
        windows={
            "scenario": (start, end),
            "train": (start, SCENARIO_TEST_START),
            "test": (SCENARIO_TEST_START, end),
        },
        files=paths,
        extras={
            "seed": args.seed,
            "mains_rows": int(len(out.mains)),
            "onset_counts": {
                str(name): int((out.onsets["appliance"] == name).sum())
                for name in out.per_appliance.keys()
            },
        },
    )
    logger.info("generate_scenario: done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
