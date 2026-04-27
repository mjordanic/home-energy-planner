"""NILM training — placeholder.

NILM disaggregation is not the focus of this project. The default
:class:`~aerogrid.nilm.disaggregator.Disaggregator` returns ground-truth
per-appliance power from the simulator and requires no training.

If you plug in a real :class:`~aerogrid.nilm.model.DisaggModel`, add your
training logic here and invoke it from ``scripts/train_disaggregator.py``.
"""
