"""Scoring harness for llm-systems-cookbook notebooks.

Every notebook instantiates a :class:`Scorer` in its second cell and calls
``s.check(...)`` / ``s.assert_close(...)`` / ``s.benchmark(...)`` to record
numerical checks. The final cell calls ``s.summary()`` and ``s.save()`` which
writes ``scores/{notebook_id}.json`` to the repository root.
"""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCORES_DIR = REPO_ROOT / "scores"
SCORES_DIR.mkdir(exist_ok=True)


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str = ""
    duration_s: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    skipped: bool = False


class Scorer:
    """Collects scored checks for a notebook and persists results as JSON."""

    def __init__(self, notebook_id: str) -> None:
        self.notebook_id = notebook_id
        self.results: list[CheckResult] = []
        self._start = time.time()

    def check(
        self,
        name: str,
        predicate: Callable[[], bool],
        msg: str = "",
    ) -> bool:
        t0 = time.time()
        try:
            ok = bool(predicate())
            r = CheckResult(
                name,
                ok,
                msg if ok else f"predicate False ({msg})" if msg else "predicate False",
                time.time() - t0,
            )
        except Exception as e:  # noqa: BLE001 - harness intentionally logs every failure
            r = CheckResult(
                name,
                False,
                f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}",
                time.time() - t0,
            )
        self.results.append(r)
        print(f"{'PASS' if r.passed else 'FAIL'}  {name}  {r.message}".strip())
        return r.passed

    def assert_close(
        self,
        name: str,
        actual: float,
        expected: float,
        rtol: float = 0.05,
    ) -> bool:
        t0 = time.time()
        denom = max(abs(expected), 1e-12)
        rel = abs(actual - expected) / denom
        ok = rel <= rtol
        r = CheckResult(
            name,
            ok,
            (f"actual={actual:.4g} expected={expected:.4g} rel_err={rel:.3%} tol={rtol:.1%}"),
            duration_s=time.time() - t0,
            details={
                "actual": float(actual),
                "expected": float(expected),
                "rel_err": float(rel),
                "rtol": float(rtol),
            },
        )
        self.results.append(r)
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {r.message}")
        return ok

    def benchmark(
        self,
        name: str,
        measured: float,
        baseline: float | None = None,
        must_beat: float = 1.0,
        unit: str = "",
        higher_is_better: bool = True,
    ) -> bool:
        if baseline is None:
            r = CheckResult(
                name,
                True,
                f"{measured:.3g} {unit}".strip(),
                details={"measured": float(measured), "unit": unit},
            )
        else:
            ratio = (measured / baseline) if higher_is_better else (baseline / measured)
            ok = ratio >= must_beat
            r = CheckResult(
                name,
                ok,
                (
                    f"{measured:.3g} vs baseline {baseline:.3g} {unit} "
                    f"ratio={ratio:.2f}x must_beat={must_beat}x"
                ).strip(),
                details={
                    "measured": float(measured),
                    "baseline": float(baseline),
                    "ratio": float(ratio),
                    "must_beat": float(must_beat),
                    "unit": unit,
                    "higher_is_better": higher_is_better,
                },
            )
        self.results.append(r)
        print(f"{'PASS' if r.passed else 'FAIL'}  {name}  {r.message}")
        return r.passed

    def skip(self, name: str, reason: str) -> None:
        """Record a skipped check. Skipped entries are not counted toward total or passed."""
        r = CheckResult(name, False, f"skipped: {reason}", 0.0, skipped=True)
        self.results.append(r)
        print(f"SKIP  {name}  {reason}")

    def _build_summary(self) -> dict[str, Any]:
        scored = [r for r in self.results if not r.skipped]
        total = len(scored)
        # Coerce to native int so ``sum`` of numpy bools doesn't produce
        # numpy scalars that json.dumps then stringifies via default=str.
        passed = int(sum(1 for r in scored if r.passed))
        return {
            "notebook_id": self.notebook_id,
            "passed": passed,
            "total": total,
            "skipped": sum(1 for r in self.results if r.skipped),
            "score": float(passed / total) if total else 0.0,
            "elapsed_s": time.time() - self._start,
            "results": [asdict(r) for r in self.results],
        }

    def summary(self) -> dict[str, Any]:
        out = self._build_summary()
        print(
            f"\n[{self.notebook_id}] {out['passed']}/{out['total']} checks passed "
            f"({out['score']:.0%}); skipped={out['skipped']}"
        )
        return out

    def save(self) -> Path:
        # Build the summary without printing; the notebook will have
        # already called summary() explicitly. Calling save() alone also
        # works - nothing prints twice in either ordering.
        data = self._build_summary()
        path = SCORES_DIR / f"{self.notebook_id}.json"
        path.write_text(json.dumps(data, indent=2, default=str))
        return path
