"""One-call notebook bootstrap.

Every notebook's second cell collapses to::

    from llm_systems_cookbook.nb import bootstrap
    s = bootstrap("<track>_<NN>_<slug>")

:func:`bootstrap` sets the seed, prints a hardware summary, ensures the
``scoring`` package (which lives at the repo root and is not pip-installed)
is importable, and writes ``DEVICE`` and ``IS_CUDA`` into the caller's
globals so subsequent cells can use them unchanged.
"""

from __future__ import annotations

import inspect
import sys
from typing import TYPE_CHECKING, Any

from ._utils import hardware_check, repo_root, set_seed

if TYPE_CHECKING:
    from scoring.harness import Scorer


def _ensure_scoring_on_path() -> None:
    root = str(repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def bootstrap(
    notebook_id: str,
    *,
    seed: int = 0,
    print_hardware: bool = True,
) -> Scorer:
    """Initialise a notebook: seed RNGs, print hardware, return a ``Scorer``.

    Side effects: prepends the repo root to ``sys.path`` (idempotent),
    prints the output of :func:`hardware_check` when ``print_hardware`` is
    true, and — when ``torch`` is importable — injects ``DEVICE`` (a
    ``torch.device``) and ``IS_CUDA`` (a ``bool``) into the calling
    notebook's globals.
    """

    _ensure_scoring_on_path()
    from scoring.harness import Scorer  # noqa: PLC0415 - after sys.path fix

    set_seed(seed)
    if print_hardware:
        print(hardware_check())

    scorer = Scorer(notebook_id)

    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return scorer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    caller_globals: dict[str, Any] = inspect.stack()[1].frame.f_globals
    caller_globals["DEVICE"] = device
    caller_globals["IS_CUDA"] = device.type == "cuda"
    return scorer
