"""System information and CPU-usage measurement built on psutil.

CPU utilisation semantics (kept explicit to avoid over-claiming):
* ``system_cpu_percent`` – aggregate, machine-wide utilisation averaged over all logical
  CPUs during the measured interval (0-100 %). It includes unrelated background load and
  is NOT a per-core measurement.
* ``process_cpu_time_s`` – user+system CPU seconds consumed by the benchmarking process
  and its worker processes during the interval.
* ``effective_cores_used`` – process_cpu_time_s / wall time: average number of logical
  CPUs kept busy by this benchmark. On Windows CPU-time resolution is ~15.6 ms, so this
  value is unreliable for very short runs.
* Intervals shorter than ``MIN_RELIABLE_INTERVAL_S`` are reported as NaN (not measurable)
  for both utilisation figures instead of a misleading 0.
"""
from __future__ import annotations

import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import psutil

MIN_RELIABLE_INTERVAL_S = 0.05


@lru_cache(maxsize=1)
def cpu_model_name() -> str:
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=15, check=True,
            ).stdout.strip()
            if out:
                return out.splitlines()[0].strip()
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"[system_monitor] could not query CPU name via PowerShell: {exc}", file=sys.stderr)
    return platform.processor() or "unknown"


def get_system_info() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    freq = psutil.cpu_freq()
    return {
        "cpu_model": cpu_model_name(),
        "physical_cores": psutil.cpu_count(logical=False) or 1,
        "logical_cores": psutil.cpu_count(logical=True) or 1,
        "cpu_max_freq_mhz": round(freq.max, 1) if freq and freq.max else None,
        "ram_total_gb": round(vm.total / 1024**3, 2),
        "ram_available_gb": round(vm.available / 1024**3, 2),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python_version": platform.python_version(),
    }


def current_cpu_percent(interval: float = 0.5, percpu: bool = False) -> float | list[float]:
    return psutil.cpu_percent(interval=interval, percpu=percpu)


@dataclass
class CpuUsage:
    system_cpu_percent: float
    process_cpu_time_s: float
    effective_cores_used: float


class CpuSampler:
    """Measures CPU usage of the current process plus given worker PIDs over an interval."""

    def __init__(self, extra_pids: list[int] | None = None):
        self._procs = [psutil.Process()]
        for pid in extra_pids or []:
            self._procs.append(psutil.Process(pid))
        self._cpu0 = 0.0
        self._t0 = 0.0

    def _total_cpu(self) -> float:
        total = 0.0
        for p in self._procs:
            t = p.cpu_times()
            total += t.user + t.system
        return total

    def start(self) -> None:
        psutil.cpu_percent(interval=None)  # resets the system-wide counter
        self._cpu0 = self._total_cpu()
        self._t0 = time.perf_counter()

    def stop(self) -> CpuUsage:
        wall = time.perf_counter() - self._t0
        system = psutil.cpu_percent(interval=None)
        cpu = max(0.0, self._total_cpu() - self._cpu0)
        if wall < MIN_RELIABLE_INTERVAL_S:
            return CpuUsage(float("nan"), cpu, float("nan"))
        return CpuUsage(system_cpu_percent=float(system), process_cpu_time_s=cpu,
                        effective_cores_used=cpu / wall)
