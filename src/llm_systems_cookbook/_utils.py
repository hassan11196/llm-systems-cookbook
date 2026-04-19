"""Shared utilities used across notebooks.

Keep this module import-light: notebooks import it in cell 2 and expect it to
succeed without any heavy downloads or CUDA initialisation side effects.
"""

from __future__ import annotations

import contextlib
import os
import random
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# When a notebook is executed headlessly (e.g. `jupyter nbconvert --execute`
# under CI, Sphinx/MyST-NB, or a display-less Binder runner) matplotlib
# falls back to the default interactive backend and its figure window
# blocks forever. Force the Agg backend whenever there's no DISPLAY so
# `plt.show()` becomes a no-op in batch contexts while staying inline
# when a GUI is available. Safe to set before matplotlib is imported.
if os.environ.get("DISPLAY", "") == "" and "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"


def set_seed(seed: int = 0) -> None:
    """Seed Python, NumPy, and Torch RNGs deterministically.

    Does not call ``torch.use_deterministic_algorithms(True)`` because that
    flag conflicts with some SDPA and Triton kernels used later in the
    curriculum. Notebooks that need strict determinism should call
    :func:`strict_determinism` explicitly.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np  # noqa: PLC0415 - imported lazily to keep _utils import cheap
    except ImportError:
        pass
    else:
        np.random.seed(seed)

    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def strict_determinism() -> None:
    """Opt-in flag for notebooks that need bit-reproducibility.

    Warning: this can raise on operations that lack deterministic kernels
    (e.g., scaled_dot_product_attention, several Triton paths).
    """

    import torch  # noqa: PLC0415

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> Any:
    """Return ``torch.device('cuda')`` if CUDA is available, else CPU."""

    import torch  # noqa: PLC0415

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hardware_check(min_cc: tuple[int, int] | None = None) -> dict[str, str | int | float]:
    """Describe the active device and (optionally) gate on a compute-capability floor.

    When CUDA is unavailable, returns a CPU descriptor. When ``min_cc`` is
    provided and the device's compute capability is below it, raises
    ``SystemExit`` with a message telling the user what hardware is required.
    """

    try:
        import torch  # noqa: PLC0415
    except ImportError as e:
        raise SystemExit(f"torch is required: {e}") from e

    info: dict[str, str | int | float] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": int(torch.cuda.is_available()),
    }

    if not torch.cuda.is_available():
        info["device"] = "cpu"
        if min_cc is not None:
            raise SystemExit(
                f"This notebook requires CUDA compute capability >= {min_cc}. "
                "No CUDA device detected. On Colab, switch runtime to GPU; locally, "
                "ensure a compatible NVIDIA GPU and CUDA toolkit are installed."
            )
        return info

    props = torch.cuda.get_device_properties(0)
    cc = torch.cuda.get_device_capability(0)
    info.update(
        {
            "device": props.name,
            "compute_capability": f"{cc[0]}.{cc[1]}",
            "total_memory_gb": round(props.total_memory / 1024**3, 2),
            "multi_processor_count": props.multi_processor_count,
        }
    )
    if min_cc is not None and cc < min_cc:
        raise SystemExit(
            f"This notebook requires compute capability >= {min_cc}. "
            f"Detected {cc} on {props.name}. See the notebook header for a fallback path."
        )
    return info


@contextlib.contextmanager
def timeit(label: str = "", sync_cuda: bool = False) -> Iterator[dict[str, float]]:
    """Context manager timing a block with ``perf_counter_ns``.

    Yields a dict that is populated with ``elapsed_s`` on exit. Pass
    ``sync_cuda=True`` to call ``torch.cuda.synchronize()`` at both ends so
    GPU-kernel timings reflect wall-clock latency.
    """

    torch = None
    if sync_cuda:
        import torch as _torch  # noqa: PLC0415

        if _torch.cuda.is_available():
            torch = _torch
            torch.cuda.synchronize()
    result: dict[str, float] = {}
    start = time.perf_counter_ns()
    try:
        yield result
    finally:
        if torch is not None:
            torch.cuda.synchronize()
        elapsed = (time.perf_counter_ns() - start) / 1e9
        result["elapsed_s"] = elapsed
        if label:
            print(f"[timeit] {label}: {elapsed * 1000:.2f} ms")


def repo_root() -> Path:
    """Path to the repo root — handy for notebooks regardless of cwd."""

    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """Return (and create) the repo-level ``data/`` directory."""

    p = repo_root() / "data"
    p.mkdir(exist_ok=True)
    return p


def download_wikitext2(split: str = "test") -> Path:
    """Download the wikitext-2 raw split into ``data/wikitext-2-raw-v1/``.

    Uses the HF ``datasets`` library. Cached across invocations.
    """

    from datasets import load_dataset  # noqa: PLC0415

    cache_dir = data_dir() / ".hf_cache"
    cache_dir.mkdir(exist_ok=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split, cache_dir=str(cache_dir))
    out_path = data_dir() / f"wikitext-2-raw-v1-{split}.txt"
    if not out_path.exists():
        text = "\n".join(row["text"] for row in ds if row["text"].strip())
        out_path.write_text(text)
    return out_path
