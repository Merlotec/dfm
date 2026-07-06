"""
FVM dataset for DFM training.

Each item is a [T, C, H, W] tensor of T consecutive normalised, rendered frames
from one simulation run.  T must be >= n_warmup_frames + 2 so the trainer has
enough frames for warmup + one training step.

The renderer (MeshRenderer) converts FVM cell-level primitives to a smooth
pixel grid via barycentric interpolation.  It is built once per dataset and
cached to disk alongside the data as renderer_cache_{H}x{W}.pt.
"""

import json
import os
import sys
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# ---- inject solver path so MeshRenderer is importable ----
_FLSIM_ROOT = Path(__file__).resolve().parents[2]  # .../flsim
_SOLVER_DIR = _FLSIM_ROOT / 'fvm_solver'
_GEN_DIR    = _FLSIM_ROOT / 'fvm_model' / 'fvm_gen'
for _p in (_SOLVER_DIR, _GEN_DIR):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from renderer import MeshRenderer  # noqa: E402  (needs path injection above)

# Back-compat: shared_mesh.pkl predates the solver reorganisation that moved
# FVMMesh from `time_fvm.fvm_mesh` to `time_fvm.mesh_utils.fvm_mesh`.  Alias the
# old module path so the pickle resolves.
try:
    import time_fvm.mesh_utils.fvm_mesh as _fvm_mesh_mod  # noqa: E402
    sys.modules.setdefault('time_fvm.fvm_mesh', _fvm_mesh_mod)
except Exception:
    pass


def _mesh_triangles(fvm_mesh):
    """Cell→vertex connectivity, renamed `cells` → `triangles` post-reorg."""
    tris = getattr(fvm_mesh, 'triangles', None)
    if tris is None:
        tris = fvm_mesh.cells
    return tris


# ---------------------------------------------------------------------------
# Renderer factory
# ---------------------------------------------------------------------------

def build_renderer(dataset_dir: Path, resolution: tuple[int, int],
                   device: str = 'cpu') -> MeshRenderer:
    """Load renderer from cache if available and consistent, otherwise build and cache it."""
    H, W = resolution
    cache = dataset_dir / f'renderer_cache_{H}x{W}.pt'

    mesh_pkl = dataset_dir / 'shared_mesh.pkl'
    if not mesh_pkl.exists():
        raise FileNotFoundError(f'shared_mesh.pkl not found in {dataset_dir}')
    with open(mesh_pkl, 'rb') as f:
        mesh_dict = pickle.load(f)
    fvm_mesh = mesh_dict['mesh']
    tris  = _mesh_triangles(fvm_mesh)
    verts = fvm_mesh.vertices.cpu().numpy()
    n_cells = int(tris.shape[0])
    x0, x1 = float(verts[:, 0].min()), float(verts[:, 0].max())
    y0, y1 = float(verts[:, 1].min()), float(verts[:, 1].max())

    if cache.exists():
        renderer = MeshRenderer.from_cache(str(cache), device=device)
        eps = 1e-3
        cache_ok = (
            renderer._c2v_tri.max().item() + 1 == n_cells
            and abs(renderer.xlim[0] - x0) < eps and abs(renderer.xlim[1] - x1) < eps
            and abs(renderer.ylim[0] - y0) < eps and abs(renderer.ylim[1] - y1) < eps
        )
        if cache_ok:
            return renderer
        print('  Renderer cache stale (mesh mismatch), rebuilding...')
        cache.unlink()

    renderer = MeshRenderer(
        verts,
        tris.cpu().numpy(),
        resolution=resolution,
        device=device,
    )
    renderer.save_cache(str(cache))
    return renderer


# ---------------------------------------------------------------------------
# Pixel mask (fluid vs. hole)
# ---------------------------------------------------------------------------

def pad_to_multiple(x: torch.Tensor, m: int, value: float = 0.0) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad the last two dims of x up to a multiple of `m` (bottom/right).

    Returns (padded, (H_orig, W_orig)) — the original size is what tells you where the
    pad begins (everything at row >= H_orig or col >= W_orig is padding / outside frame).
    """
    H, W = x.shape[-2], x.shape[-1]
    ph, pw = (-H) % m, (-W) % m
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), value=value)
    return x, (H, W)


def build_pixel_mask(renderer: MeshRenderer, resolution: tuple[int, int]) -> torch.Tensor:
    """Boolean (1, 1, H, W) mask — True for pixels inside the fluid mesh."""
    H, W = resolution
    mask = torch.zeros(H * W, dtype=torch.bool)
    mask[renderer._interior_idx] = True
    return mask.view(1, 1, H, W)


def load_pixel_mask(dataset_dir: Path, renderer: MeshRenderer,
                    resolution: tuple[int, int], frame_mask: bool = False,
                    native_hw: Optional[tuple[int, int]] = None) -> torch.Tensor:
    """Return the cached is_valid pixel mask (1, 1, H, W).

    When ``frame_mask`` is set, return a 2-channel (1, 2, H, W) mask
    ``[is_valid, is_inside_frame]``: is_valid is True on fluid pixels only, and
    is_inside_frame is False on the pad border (rows >= native_hw[0] or cols >=
    native_hw[1]) so the model can tell padding apart from colliders.  If native_hw
    is None the frame is assumed to fill the grid (is_inside_frame all True).
    """
    H, W = resolution
    cache = dataset_dir / f'pixel_mask_{H}x{W}.pt'
    n_interior = len(renderer._interior_idx)
    if cache.exists():
        valid = torch.load(cache, weights_only=True)
        if not (valid.shape == (1, 1, H, W)
                and int(valid.sum()) == n_interior
                and valid.view(-1)[renderer._interior_idx].all()):
            print('  Pixel mask stale (renderer mismatch), rebuilding...')
            cache.unlink()
            valid = None
    else:
        valid = None
    if valid is None:
        valid = build_pixel_mask(renderer, resolution)
        torch.save(valid, cache)
        print(f'  Pixel mask saved — {valid.sum().item()} fluid / {valid.numel()} total pixels')

    if not frame_mask:
        return valid
    inside = torch.ones_like(valid)
    if native_hw is not None:
        h0, w0 = native_hw
        inside[..., h0:, :] = False
        inside[..., :, w0:] = False
        valid = valid & inside                         # fluid can't be in the pad region
    return torch.cat([valid, inside], dim=1)           # [1, 2, H, W]


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def compute_normalisation_stats(
    sim_dirs: list[Path],
    renderer: MeshRenderer,
    n_samples: int = 300,
    first_frame: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate per-channel mean and std by sampling frames across all runs."""
    all_files: list[Path] = []
    for d in sim_dirs:
        all_files.extend(sorted(
            [f for f in d.iterdir() if f.name.startswith('t_') and f.name.endswith('.npz')],
            key=lambda f: float(f.stem[2:]),
        )[first_frame:])

    n = min(n_samples, len(all_files))
    idx = torch.randperm(len(all_files))[:n].tolist()

    C = 4
    s1 = torch.zeros(C)
    s2 = torch.zeros(C)
    cnt = torch.zeros(C)
    for i in idx:
        d = np.load(all_files[i])
        vals = d['cell_primatives'].astype(np.float32) * d['prim_std'] + d['prim_mean']
        frame = renderer.render_cell_smooth(vals)   # [C, H, W]
        for c in range(C):
            px = frame[c]
            fin = px[torch.isfinite(px)]
            s1[c]  += fin.sum()
            s2[c]  += (fin ** 2).sum()
            cnt[c] += fin.numel()

    mean = s1 / cnt
    std  = ((s2 / cnt) - mean ** 2).clamp(min=0).sqrt().clamp(min=1e-6)
    return mean, std


# ---------------------------------------------------------------------------
# Single-run dataset
# ---------------------------------------------------------------------------

class FVMSequenceDataset(Dataset):
    """
    Prediction windows and decoupled context, both drawn from one simulation run.

    Each item is a tuple ``(context, pred)``:
      - ``pred``    : [pred_len, C, H, W]   consecutive frames = seed + targets,
                      a sliding window over the run (the quantity being predicted).
      - ``context`` : [n_context, C, H, W]  a contiguous block of frames sampled
                      from a *random* offset in the SAME run, independent of the
                      prediction window.

    Decoupling context from the prediction window forces the ContextEncoder to
    summarise the run's governing dynamics (viscosity, BCs, geometry) rather than
    a snapshot of the state immediately preceding the seed.  Set
    ``random_context=False`` (eval) to take the context deterministically from the
    start of the run.
    """

    def __init__(
        self,
        sim_dir:        Path,
        renderer:       MeshRenderer,
        n_context:      int,
        pred_len:       int,
        mean:           torch.Tensor,
        std:            torch.Tensor,
        first_frame:    int = 20,
        frame_cache:    Optional[list[torch.Tensor]] = None,
        random_context: bool = True,
    ):
        files = sorted(
            [f for f in sim_dir.iterdir() if f.name.startswith('t_') and f.name.endswith('.npz')],
            key=lambda f: float(f.stem[2:]),
        )[first_frame:]
        self.paths          = files
        self.renderer       = renderer
        self.n_context      = n_context
        self.pred_len       = pred_len
        self.random_context = random_context
        self.mean           = mean.view(-1, 1, 1)   # [C, 1, 1] for broadcasting
        self.std            = std.view(-1, 1, 1)
        self._cache         = frame_cache            # optional pre-rendered cache

    def __len__(self) -> int:
        # Need enough frames for both a prediction window and a context block.
        if len(self.paths) < self.n_context:
            return 0
        return max(0, len(self.paths) - self.pred_len + 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pred = torch.stack([self._get_frame(idx + i) for i in range(self.pred_len)])

        n = len(self.paths)
        if self.random_context:
            c_start = int(torch.randint(0, n - self.n_context + 1, (1,)).item())
        else:
            c_start = 0
        context = torch.stack([self._get_frame(c_start + i) for i in range(self.n_context)])

        return context, pred   # [n_context, C, H, W], [pred_len, C, H, W]

    def _get_frame(self, i: int) -> torch.Tensor:
        if self._cache is not None:
            return self._cache[i]
        d    = np.load(self.paths[i])
        vals = d['cell_primatives'].astype(np.float32) * d['prim_std'] + d['prim_mean']
        raw  = self.renderer.render_cell_smooth(vals)   # [C, H, W]
        return (raw - self.mean) / self.std

    @classmethod
    def with_cache(cls, sim_dir: Path, renderer: MeshRenderer,
                   n_context: int, pred_len: int,
                   mean: torch.Tensor, std: torch.Tensor,
                   first_frame: int = 20,
                   random_context: bool = True) -> "FVMSequenceDataset":
        """Pre-render and cache all frames in memory for fast repeated access."""
        files = sorted(
            [f for f in sim_dir.iterdir() if f.name.startswith('t_') and f.name.endswith('.npz')],
            key=lambda f: float(f.stem[2:]),
        )[first_frame:]
        m = mean.view(-1, 1, 1)
        s = std.view(-1, 1, 1)
        cache = []
        for path in files:
            d    = np.load(path)
            vals = d['cell_primatives'].astype(np.float32) * d['prim_std'] + d['prim_mean']
            raw  = renderer.render_cell_smooth(vals)
            cache.append((raw - m) / s)
        return cls(sim_dir, renderer, n_context, pred_len, mean, std,
                   first_frame, frame_cache=cache, random_context=random_context)


# ---------------------------------------------------------------------------
# Multi-run data module (plain PyTorch, no Lightning dependency)
# ---------------------------------------------------------------------------

class _IndexedDataset(Dataset):
    """Wraps a dataset so each item is (global_index, *item) — for latent caching."""

    def __init__(self, ds: Dataset):
        self.ds = ds

    def __len__(self) -> int:
        return len(self.ds)  # type: ignore[arg-type]

    def __getitem__(self, i: int):
        item = self.ds[i]
        return (i, *item) if isinstance(item, tuple) else (i, item)


class FVMDataModule:
    """
    Scans a dataset directory for simulation subdirectories, builds a renderer,
    computes normalisation statistics, and exposes a DataLoader.

    Usage
    -----
    dm = FVMDataModule(data_dir, n_context=5, horizon=4, batch_size=4)
    dm.setup()
    for context, pred in dm.train_dataloader():
        context_frames = [context[:, t] for t in range(context.shape[1])]  # [B,C,H,W]
        pred_frames    = [pred[:, t]    for t in range(pred.shape[1])]      # [B,C,H,W]
    """

    STATS_FILE = 'hfm_input_stats.json'

    def __init__(
        self,
        data_dir:    Path,
        n_context:   int,
        horizon:     int,
        resolution:  tuple[int, int] = (256, 256),
        batch_size:  int = 4,
        num_workers: int = 4,
        first_frame: int = 20,
        cache_frames: bool = False,
        random_context: bool = True,
        return_index: bool = False,
        mean: Optional[torch.Tensor] = None,
        std:  Optional[torch.Tensor] = None,
    ):
        self.data_dir       = Path(data_dir)
        self.n_context      = n_context
        self.horizon        = horizon
        self.pred_len       = horizon + 1          # seed + horizon targets
        self.resolution     = resolution
        self.batch_size     = batch_size
        self.num_workers    = num_workers
        self.first_frame    = first_frame
        self.cache_frames   = cache_frames
        self.random_context = random_context
        self.return_index   = return_index         # yield (idx, context, pred) for latent caching
        self._dataset: Optional[ConcatDataset] = None
        self.mean: Optional[torch.Tensor] = mean
        self.std:  Optional[torch.Tensor] = std

    def setup(self, recompute_stats: bool = False):
        renderer = build_renderer(self.data_dir, self.resolution)

        # A simulation directory is any subdirectory containing frame files
        # (t_*.npz).  This is robust to run-naming conventions (run*, mu_b_*, …).
        sim_dirs = sorted(
            p for p in self.data_dir.iterdir()
            if p.is_dir() and not p.name.startswith('.')
            and any(f.name.startswith('t_') and f.name.endswith('.npz') for f in p.iterdir())
        )
        if not sim_dirs:
            raise RuntimeError(f'No simulation subdirectories found in {self.data_dir}')

        # Load or compute normalisation stats (skip if already provided externally)
        if self.mean is None or self.std is None:
            stats_path = self.data_dir / self.STATS_FILE
            if stats_path.exists() and not recompute_stats:
                with open(stats_path) as f:
                    s = json.load(f)
                self.mean = torch.tensor(s['mean'])
                self.std  = torch.tensor(s['std'])
            else:
                print('Computing normalisation stats...')
                self.mean, self.std = compute_normalisation_stats(
                    sim_dirs, renderer, first_frame=self.first_frame)
                with open(stats_path, 'w') as f:
                    json.dump({'mean': self.mean.tolist(), 'std': self.std.tolist()}, f)
                print(f'Stats saved to {stats_path}')

        builder = FVMSequenceDataset.with_cache if self.cache_frames else (
            lambda *a, **kw: FVMSequenceDataset(*a, **kw)
        )
        datasets = [
            builder(d, renderer, self.n_context, self.pred_len, self.mean, self.std,
                    self.first_frame, random_context=self.random_context)
            for d in sim_dirs
        ]
        datasets = [ds for ds in datasets if len(ds) > 0]
        if not datasets:
            raise RuntimeError('No usable sequences found — try reducing horizon or first_frame')
        self._dataset = ConcatDataset(datasets)
        print(f'Dataset ready: {len(self._dataset)} sequences across {len(datasets)} runs')

    def _num_workers(self) -> int:
        # When frames are cached in RAM, __getitem__ is a cheap index+normalise,
        # so worker processes give no speedup — and forking them would each
        # inherit the multi-GB cache, exhausting a memory-limited job (fork
        # ENOMEM).  Load in-process instead.
        return 0 if self.cache_frames else self.num_workers

    def _wrap(self, ds):
        return _IndexedDataset(ds) if self.return_index else ds

    def train_dataloader(self) -> DataLoader:
        assert self._dataset is not None, 'Call setup() first'
        w = self._num_workers()
        return DataLoader(
            self._wrap(self._dataset),
            batch_size         = self.batch_size,
            shuffle            = True,
            num_workers        = w,
            pin_memory         = True,
            persistent_workers = w > 0,
            drop_last          = True,   # constant batch shape → no last-batch recompile
        )

    def val_dataloader(self) -> DataLoader:
        assert self._dataset is not None, 'Call setup() first'
        w = self._num_workers()
        return DataLoader(
            self._wrap(self._dataset),
            batch_size         = self.batch_size,
            shuffle            = False,
            num_workers        = w,
            pin_memory         = True,
            persistent_workers = w > 0,
        )
