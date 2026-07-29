"""
Host telemetry sampling.

One sampler instance is held by the engine server and polled on a fixed
interval. Every read is defensive: a missing dependency, an absent GPU,
or a transient psutil error degrades to ``None`` fields rather than
breaking the metrics stream — a dashboard with a gap is useful, a dead
socket is not.

GPU figures come from ``nvidia-smi`` when present. The probe result is
cached: a machine without the binary is only probed once.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from typing import Any, Optional

from src.logger import get_logger

try:  # psutil is a hard requirement in requirements.txt, but the
    import psutil  # dashboard must not kill the engine if it is absent.
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

_GPU_QUERY = [
    "nvidia-smi",
    "--query-gpu=utilization.gpu,memory.used,memory.total",
    "--format=csv,noheader,nounits",
]
_GPU_TIMEOUT_SECONDS = 2.0


class MetricsSampler:
    """
    Samples host CPU, memory, GPU, and network counters.

    The sampler keeps the little state psutil needs for rate calculations
    (previous network counters) and the cached GPU availability verdict.
    """

    def __init__(self) -> None:
        """Initialize the sampler and prime psutil's CPU window."""
        self._logger = get_logger(__name__)
        self._gpu_available: Optional[bool] = None
        self._last_net: Optional[tuple[float, int, int]] = None

        if psutil is not None:
            # First cpu_percent call establishes the measurement window;
            # its return value is meaningless and discarded.
            try:
                psutil.cpu_percent(interval=None)
            except Exception:
                pass

    async def sample(self) -> dict[str, Any]:
        """
        Take one host telemetry sample.

        Returns:
            JSON-compatible metrics. Fields whose source is unavailable
            are None rather than absent, so the UI can label them.
        """
        cpu = ram = ram_available = ram_total = None
        net_sent_rate = net_recv_rate = None

        if psutil is not None:
            try:
                cpu = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                ram = memory.percent
                ram_available = memory.available
                ram_total = memory.total
            except Exception as exc:
                self._logger.debug("psutil sample failed: %s", exc)

            try:
                now = asyncio.get_running_loop().time()
                counters = psutil.net_io_counters()
                if self._last_net is not None:
                    then, sent, recv = self._last_net
                    elapsed = max(now - then, 0.001)
                    net_sent_rate = (counters.bytes_sent - sent) / elapsed
                    net_recv_rate = (counters.bytes_recv - recv) / elapsed
                self._last_net = (
                    now,
                    counters.bytes_sent,
                    counters.bytes_recv,
                )
            except Exception as exc:
                self._logger.debug("network sample failed: %s", exc)

        gpu = await self._sample_gpu()

        return {
            "cpuPercent": cpu,
            "ramPercent": ram,
            "ramAvailableBytes": ram_available,
            "ramTotalBytes": ram_total,
            "netSentBytesPerSec": net_sent_rate,
            "netRecvBytesPerSec": net_recv_rate,
            "gpu": gpu,
        }

    async def _sample_gpu(self) -> Optional[dict[str, Any]]:
        """
        Sample GPU utilization via nvidia-smi when available.

        Returns:
            GPU metrics, or None when no NVIDIA GPU is usable.
        """
        if self._gpu_available is None:
            self._gpu_available = shutil.which("nvidia-smi") is not None

        if not self._gpu_available:
            return None

        def probe() -> Optional[dict[str, Any]]:
            try:
                output = subprocess.run(
                    _GPU_QUERY,
                    capture_output=True,
                    text=True,
                    timeout=_GPU_TIMEOUT_SECONDS,
                    check=True,
                ).stdout.strip()
                first_line = output.splitlines()[0]
                util, used, total = (
                    float(part.strip()) for part in first_line.split(",")
                )
                return {
                    "utilizationPercent": util,
                    "memoryUsedMb": used,
                    "memoryTotalMb": total,
                }
            except Exception:
                return None

        result = await asyncio.to_thread(probe)
        if result is None:
            # Repeated failures mean the GPU is not usable; stop probing.
            self._gpu_available = False
        return result


__all__ = [
    "MetricsSampler",
]
