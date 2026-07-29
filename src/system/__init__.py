"""
System package: host telemetry for the Mission Control dashboard.

MetricsSampler reads CPU, memory, GPU, and network figures from the host
so the dashboard can show what research actually costs the machine.
"""

from .metrics import MetricsSampler

__all__ = [
    "MetricsSampler",
]
