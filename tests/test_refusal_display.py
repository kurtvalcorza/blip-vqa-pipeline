"""DAT19/DAT20: a refused BYOD input shows one clean line, ``Refused: <file>: <reason>``, with no traceback.

The router cell carries ml-worker's refusal channel (template ``dat20-router-refusal/1``) and Section 4 wraps
its whole BYOD branch so that every user-input problem raises ``BYODRefusal``. These tests drive both from the
notebook JSON with CI's dependencies only:

- Section 4 is executed verbatim (the review-fix harness) for each refusal path;
- the router's kernel loop is driven with a queued stub connection on every platform;
- on POSIX the real worker process runs Section 4, so the full display path is checked end to end.
"""
# ruff: noqa: E501  -- notebook source literals are matched verbatim

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

import blip_vqa_pipeline as pkg
from test_blip_vqa_colab_review_fixes import _fake_colab, _records, _section4, _zip

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "blip_vqa_colab.ipynb"
STEM = "blip_vqa"

NB = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
CELLS = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in NB["cells"]]
CODE = [i for i, c in enumerate(NB["cells"]) if c["cell_type"] == "code"]


def _cell(marker: str) -> int:
    found = [i for i in CODE if marker in CELLS[i]]
    assert len(found) == 1, (marker, found)
    return found[0]


SEC4 = _cell("USE_BYOD = False  # @param")
ROUTER = _cell("_WORKER_SOURCE = r")


def refusing_section4_source(path: str) -> str:
    src = CELLS[SEC4].replace("USE_BYOD = False", "USE_BYOD = True", 1)
    return src.replace("BYOD_PATH = ''", f"BYOD_PATH = {path!r}", 1)


def _refusal(monkeypatch, tmp_path, **kwargs) -> str:
    with pytest.raises(ValueError) as info:
        _section4(monkeypatch, tmp_path, **kwargs)
    exc = info.value
    assert getattr(exc, "dimer_refusal", False) is True, repr(exc)
    lines = exc._render_traceback_()
    assert len(lines) == 1 and lines[0].startswith("Refused: ") and "Traceback" not in lines[0], lines
    return lines[0]


# ---- each refusal path in Section 4 is one clean line naming the file ---------------------------------------


def test_acquisition_refusals(monkeypatch, tmp_path):
    _fake_colab(monkeypatch, [{}, {"a.zip": b"x", "b.zip": b"y"}, {"notes.txt": b"x"}])
    assert _refusal(monkeypatch, tmp_path).startswith("Refused: (no file): upload exactly one .zip file (received 0)")
    assert _refusal(monkeypatch, tmp_path).startswith("Refused: a.zip, b.zip: upload exactly one .zip file")
    assert _refusal(monkeypatch, tmp_path).startswith("Refused: notes.txt: Upload a .zip holding records.jsonl")
    missing = str(tmp_path / "missing.zip")
    assert _refusal(monkeypatch, tmp_path, byod_path=missing).startswith(f"Refused: {missing}: BYOD_PATH does not exist")
    monkeypatch.setitem(sys.modules, "google.colab", None)
    assert "needs Google Colab" in _refusal(monkeypatch, tmp_path)


def test_content_and_contract_refusals_name_the_file(monkeypatch, tmp_path):
    sample_data = tmp_path / "sample_data"
    sample_data.mkdir()
    (sample_data / "california_housing_test.csv").write_text("longitude,latitude\n1,2\n", encoding="utf-8")
    assert _refusal(monkeypatch, tmp_path, byod_path=str(sample_data)).startswith("Refused: sample_data: No records.jsonl or records.json found")
    hostname = tmp_path / "hostname"
    hostname.write_text("runtime\n", encoding="utf-8")
    assert "BYOD_PATH must name a .zip or a folder" in _refusal(monkeypatch, tmp_path, byod_path=str(hostname))
    _fake_colab(monkeypatch, [{"broken.zip": b"not a zip"}, {"small.zip": _zip(tmp_path, "small", 11)}, {"none.zip": _zip(tmp_path, "n", 12, records_name="")}])
    assert "not a readable zip file" in _refusal(monkeypatch, tmp_path)
    line = _refusal(monkeypatch, tmp_path)
    assert line.startswith("Refused: small.zip: ") and "the train split has 7 records" in line
    assert line.endswith("Fix the file, then select this cell and choose Runtime → Run after.")
    assert _refusal(monkeypatch, tmp_path).startswith("Refused: none.zip: No records.jsonl or records.json found in none.zip")


def test_a_refusal_removes_the_earlier_runs_data_and_results(monkeypatch, tmp_path):
    _fake_colab(monkeypatch, [{"ok.zip": _zip(tmp_path, "ok", 12)}])
    ns = _section4(monkeypatch, tmp_path)
    assert ns["train_records"] and list((tmp_path / "outputs").glob(f"{STEM}_*"))
    (tmp_path / "outputs" / f"{STEM}_adapter").mkdir()
    with pytest.raises(ValueError, match="does not exist"):
        exec(compile(refusing_section4_source(str(tmp_path / "missing.zip")), "<section 4>", "exec"), ns)
    assert not list((tmp_path / "outputs").glob(f"{STEM}_*"))
    for name in ("byod", "splits", "train_records", "val_records", "test_records", "data_source", "dataset_manifests"):
        assert name not in ns, name


def test_an_accepted_folder_still_runs_and_a_sample_path_failure_still_raises(monkeypatch, tmp_path):
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    folder = tmp_path / "folder"
    records = _records(folder, 12)
    (folder / "records.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    ns = _section4(monkeypatch, tmp_path, byod_path=str(folder))
    assert ns["data_source"] == "BYOD (folder)" and len(ns["train_records"]) == 8

    def broken(*args, **kwargs):
        raise ValueError("internal sample check failed")

    ns = {name: getattr(pkg, name) for name in pkg.__all__}
    ns.update({"os": os, "Path": Path, "fetch_annotations": broken})
    with pytest.raises(ValueError, match="internal sample check failed") as info:
        exec(compile(CELLS[SEC4], "<section 4>", "exec"), ns)
    assert not getattr(info.value, "dimer_refusal", False)


WORKER_SETUP = [f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r})", "import os", "from pathlib import Path", "from blip_vqa_pipeline import *"]
BAD_NAME, BAD_BYTES, BAD_RULE, DISCARDED = "hostname", b"runtime\n", "BYOD_PATH must name a .zip or a folder", "train_records"
_UNUSED = types


# ---- the router: a refusal crosses the worker boundary as one line; a genuine error keeps its traceback -----


def _router_namespace() -> dict:
    """The router cell's definitions (everything before it starts the worker), with a stand-in IPython."""
    src = CELLS[ROUTER]
    head = src[: src.index("\nif SKIP_INSTALL:")]
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: None
    saved = sys.modules.get("IPython")
    sys.modules["IPython"] = ipython
    try:
        ns: dict = {"__name__": "__main__", "os": os, "subprocess": subprocess, "sys": sys}
        exec(compile(head, f"<cell {ROUTER}>", "exec"), ns)
    finally:
        if saved is None:
            sys.modules.pop("IPython", None)
        else:
            sys.modules["IPython"] = saved
    return ns


class _Queued:
    def __init__(self, messages):
        self.messages = list(messages)
        self.sent = []

    def recv(self):
        return self.messages.pop(0)

    def send(self, message):
        self.sent.append(message)


def _stub_runtime(ns, messages):
    runtime = object.__new__(ns["IsolatedRuntime"])
    runtime._recv, runtime._send, runtime._display = _Queued(messages), _Queued([]), lambda *a, **k: None
    return runtime


def test_router_carries_the_refusal_channel():
    src = CELLS[ROUTER]
    assert 'if getattr(exc, "dimer_refusal", False):' in src and '_send.send(("refusal", str(exc)))' in src
    assert 'elif kind == "refusal":' in src and "class IsolatedCellRefusal(Exception):" in src


def test_router_kernel_shows_a_refusal_as_one_line_and_an_error_with_its_traceback(capsys):
    ns = _router_namespace()
    with pytest.raises(ns["IsolatedCellRefusal"]) as info:
        _stub_runtime(ns, [("refusal", "data.csv: a column is missing")]).run("cell")
    assert info.value._render_traceback_() == ["Refused: data.csv: a column is missing"]
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
    trace = "Traceback (most recent call last):\n  File x\nZeroDivisionError: division by zero\n"
    with pytest.raises(ns["IsolatedCellError"], match="ZeroDivisionError"):
        _stub_runtime(ns, [("error", trace, "ZeroDivisionError: division by zero")]).run("cell")
    assert capsys.readouterr().err == trace


@pytest.mark.skipif(sys.platform == "win32", reason="the router's worker uses pass_fds and POSIX sessions")
def test_real_worker_shows_the_section4_refusal_as_one_line_and_keeps_running(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    ns = _router_namespace()
    runtime = ns["IsolatedRuntime"](sys.executable, display=lambda *a, **k: None)
    try:
        for source in WORKER_SETUP:
            runtime.run(source)
        bad = tmp_path / BAD_NAME
        bad.write_bytes(BAD_BYTES)
        capsys.readouterr()
        with pytest.raises(ns["IsolatedCellRefusal"]) as info:
            runtime.run(refusing_section4_source(str(bad)))
        lines = info.value._render_traceback_()
        captured = capsys.readouterr()
        assert len(lines) == 1 and lines[0].startswith(f"Refused: {BAD_NAME}: "), lines
        assert BAD_RULE in lines[0], lines
        assert "Traceback" not in captured.out + captured.err, captured
        runtime.run(f"print('still running', {DISCARDED!r} in globals())")
        assert capsys.readouterr().out == "still running False\n"
        with pytest.raises(ns["IsolatedCellError"]):
            runtime.run("1 / 0")
        assert "Traceback" in capsys.readouterr().err
    finally:
        runtime.close()
