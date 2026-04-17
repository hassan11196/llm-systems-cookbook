"""Unit tests for the Scorer harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scoring.harness import CheckResult, Scorer


def _fresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, notebook_id: str) -> Scorer:
    monkeypatch.setattr("scoring.harness.SCORES_DIR", tmp_path)
    return Scorer(notebook_id)


def test_check_truthy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s = _fresh(tmp_path, monkeypatch, "test_truthy")
    ok = s.check("adds", lambda: 2 + 2 == 4)
    assert ok is True
    assert len(s.results) == 1
    assert s.results[0].passed is True


def test_check_falsy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s = _fresh(tmp_path, monkeypatch, "test_falsy")
    ok = s.check("always_false", lambda: False, msg="expected True")
    assert ok is False
    assert s.results[0].passed is False
    assert "expected True" in s.results[0].message


def test_check_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s = _fresh(tmp_path, monkeypatch, "test_exception")

    def _boom() -> bool:
        raise RuntimeError("nope")

    ok = s.check("raises", _boom)
    assert ok is False
    assert "RuntimeError" in s.results[0].message


def test_assert_close_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s = _fresh(tmp_path, monkeypatch, "test_close")
    # Within tolerance: passes.
    ok = s.assert_close("within_tol", actual=1.02, expected=1.0, rtol=0.05)
    assert ok is True
    # Above tolerance: fails.
    ok = s.assert_close("above_tol", actual=1.10, expected=1.0, rtol=0.05)
    assert ok is False
    details = s.results[-1].details
    assert pytest.approx(details["rel_err"], rel=1e-6) == 0.10
    # Zero expected falls back to a tiny denominator (large rel_err on any nonzero actual).
    ok = s.assert_close("zero_expected", actual=1e-6, expected=0.0, rtol=0.05)
    assert ok is False


def test_benchmark_no_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s = _fresh(tmp_path, monkeypatch, "test_bench_nb")
    ok = s.benchmark("measure", measured=42.0, unit="tok/s")
    assert ok is True
    assert "42" in s.results[0].message


def test_benchmark_with_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s = _fresh(tmp_path, monkeypatch, "test_bench_b")
    ok_hi = s.benchmark(
        "fast",
        measured=100.0,
        baseline=40.0,
        must_beat=2.0,
        unit="tok/s",
        higher_is_better=True,
    )
    assert ok_hi is True
    ok_lo = s.benchmark(
        "slow",
        measured=100.0,
        baseline=40.0,
        must_beat=2.0,
        unit="tok/s",
        higher_is_better=False,
    )
    assert ok_lo is False


def test_skip_does_not_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s = _fresh(tmp_path, monkeypatch, "test_skip")
    s.check("pass", lambda: True)
    s.skip("unavailable", "no gpu")
    summary = s.summary()
    assert summary["total"] == 1
    assert summary["passed"] == 1
    assert summary["skipped"] == 1


def test_save_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s = _fresh(tmp_path, monkeypatch, "test_save")
    s.check("one", lambda: True)
    s.assert_close("two", 1.0, 1.0)
    path = s.save()
    assert path.exists()
    assert path.name == "test_save.json"
    data = json.loads(path.read_text())
    assert data["notebook_id"] == "test_save"
    assert data["total"] == 2
    assert data["passed"] == 2
    assert data["score"] == 1.0
    assert len(data["results"]) == 2


def test_check_result_defaults() -> None:
    r = CheckResult(name="x", passed=True)
    assert r.message == ""
    assert r.details == {}
    assert r.skipped is False
