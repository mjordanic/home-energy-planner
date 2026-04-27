"""NILM disaggregation package.

NILM disaggregation is **not** the focus of this project.  The package ships
a :class:`Disaggregator` that returns ground-truth per-appliance power from
the simulator — perfect by construction.  To plug in a real NILM model,
subclass :class:`DisaggregatorBase` and/or :class:`DisaggModel`.
"""
from aerogrid.nilm.disaggregator import (
    Disaggregator,
    DisaggregatorBase,
    RollingDisaggregator,
    power_to_onsets,
)
from aerogrid.nilm.model import DisaggModel, PerfectDisaggModel
from aerogrid.nilm.onset_detector import OnsetDetector

__all__ = [
    "Disaggregator",
    "DisaggregatorBase",
    "DisaggModel",
    "OnsetDetector",
    "PerfectDisaggModel",
    "RollingDisaggregator",
    "power_to_onsets",
]
