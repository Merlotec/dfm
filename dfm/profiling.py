"""
Lightweight training-loop profiling.

`LoopProfiler` (always-on, negligible overhead) reports the two numbers that
answer "is the hardware being used well and what's the bottleneck":
  - throughput (it/s, samples/s)
  - data-wait %  — fraction of wall-clock spent waiting on the dataloader
                   (high → GPU starved by data; low → compute-bound)
  - GPU peak memory (vs total → headroom for bigger batch) and SM utilisation.

`make_profiler`/`finish_profiler` wrap torch.profiler for a `--profile N` run that
prints a kernel-level breakdown and a chrome trace.
"""

import time
from pathlib import Path

import torch


class LoopProfiler:
    def __init__(self, device: torch.device):
        self.device = device
        self.cuda = device.type == 'cuda'
        self._reset()
        self._t = time.perf_counter()

    def _reset(self):
        self.data_t = 0.0
        self.step_t = 0.0
        self.steps = 0
        self.samples = 0

    def data_ready(self):
        """Call right after a batch is fetched (records dataloader wait)."""
        now = time.perf_counter()
        self.data_t += now - self._t
        self._t = now

    def step_done(self, batch_size: int):
        """Call right after the optimizer step (records compute time)."""
        if self.cuda:
            torch.cuda.synchronize()
        now = time.perf_counter()
        self.step_t += now - self._t
        self._t = now
        self.steps += 1
        self.samples += batch_size

    def line(self) -> str:
        tot = self.data_t + self.step_t
        its = self.steps / tot if tot > 0 else 0.0
        sps = self.samples / tot if tot > 0 else 0.0
        data_pct = 100.0 * self.data_t / tot if tot > 0 else 0.0
        extra = ''
        if self.cuda:
            peak  = torch.cuda.max_memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(self.device).total_memory / 1e9
            extra = f'  mem={peak:.1f}/{total:.0f}GB'
            try:
                extra += f'  gpu={torch.cuda.utilization(self.device)}%'
            except Exception:
                pass
            torch.cuda.reset_peak_memory_stats()
        s = f'{its:.2f} it/s  {sps:.0f} samp/s  data-wait={data_pct:.0f}%{extra}'
        self._reset()
        self._t = time.perf_counter()
        return s


def make_profiler(enabled: bool, device: torch.device):
    if not enabled:
        return None
    from torch.profiler import profile, ProfilerActivity
    acts = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        acts.append(ProfilerActivity.CUDA)
    prof = profile(activities=acts, profile_memory=True, record_shapes=False)
    prof.start()
    return prof


def finish_profiler(prof, device: torch.device, out_dir) -> None:
    if prof is None:
        return
    if device.type == 'cuda':
        torch.cuda.synchronize()
    prof.stop()
    sort = 'cuda_time_total' if device.type == 'cuda' else 'cpu_time_total'
    print(f'\n===== torch.profiler: top ops by {sort} =====')
    print(prof.key_averages().table(sort_by=sort, row_limit=20))
    try:
        out = Path(out_dir) / 'profile_trace.json'
        prof.export_chrome_trace(str(out))
        print(f'chrome trace → {out}  (open in perfetto.dev or chrome://tracing)')
    except Exception as e:
        print(f'(trace export failed: {e})')
