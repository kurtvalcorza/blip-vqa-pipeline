import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blip_vqa_pipeline import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_WEIGHTS_DIR,
    IMAGE_SIZE,
    MAX_IMAGE_SIDE,
    MAX_NEW_TOKENS,
    MAX_QUESTION_CHARS,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    BlipVQAPipeline,
    stage_missing_files,
    verify_snapshot,
)

HEX40 = re.compile(r"^[0-9a-f]{40}$")
REPO = Path(__file__).resolve().parents[1]


def test_identity_constants():
    assert HEX40.match(MODEL_REVISION)
    assert MODEL_ID == "Salesforce/blip-vqa-base"
    assert DEFAULT_WEIGHTS_DIR == REPO / "weights" / MODEL_KEY
    assert 1 <= DEFAULT_MAX_NEW_TOKENS <= MAX_NEW_TOKENS == 32
    assert IMAGE_SIZE == 384 and MAX_QUESTION_CHARS == 256
    manifest = REPO / "weights" / MODEL_KEY / "dimer-base-manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        assert data["modelId"] == MODEL_ID
        assert data["revision"] == MODEL_REVISION


def _write_snapshot(root: Path, content: bytes, sha: str | None = None, size: int | None = None) -> None:
    (root / "config.json").write_bytes(content)
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {
                "path": "config.json",
                "bytes": len(content) if size is None else size,
                "sha256": hashlib.sha256(content).hexdigest() if sha is None else sha,
            }
        ],
        "totalBytes": len(content),
    }
    (root / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_verify_snapshot_accepts_matching_manifest(tmp_path):
    _write_snapshot(tmp_path, b'{"model_type": "blip"}')
    info = verify_snapshot(tmp_path)
    assert info["revision"] == MODEL_REVISION and info["files"] == 1


def test_verify_snapshot_rejects_tampered_digest(tmp_path):
    content = b'{"model_type": "blip"}'
    good = hashlib.sha256(content).hexdigest()
    flipped = ("0" if good[0] != "0" else "1") + good[1:]
    _write_snapshot(tmp_path, content, sha=flipped)
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_wrong_size_missing_file_and_revision(tmp_path):
    _write_snapshot(tmp_path, b"abc", size=99)
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, b"abc")
    manifest = json.loads((tmp_path / "dimer-base-manifest.json").read_text())
    manifest["revision"] = "0" * 40
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, b"abc")
    (tmp_path / "config.json").unlink()
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path)


def test_stage_missing_files_fetches_only_absent_entries_then_verifies(tmp_path):
    """Fresh-clone shape: manifest committed, weight file absent. allow_download fetches exactly that file."""
    payload = b"weights-bytes"
    (tmp_path / "config.json").write_bytes(b"{}")
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {"path": "config.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
            {"path": "model.bin", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
        ],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched = []

    def fake_download(relative_path, root):
        fetched.append(relative_path)
        (root / relative_path).write_bytes(payload)

    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == ["model.bin"]
    assert fetched == ["model.bin"]
    listed = verify_snapshot(tmp_path)["files"]
    assert (listed if isinstance(listed, int) else len(listed)) == 2
    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == []


def test_stage_missing_files_refuses_foreign_manifest(tmp_path):
    manifest = {"modelId": "someone/else", "revision": MODEL_REVISION, "files": []}
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)


DOCTAGS = (
    "<doctag><section_header_level_1><loc_32><loc_25><loc_231><loc_39>Report</section_header_level_1>\n"
    "<text><loc_32><loc_40><loc_319><loc_93>quarterly revenue by region</text>\n"
    "<otsl><loc_32><loc_151><loc_459><loc_303><ched>Region<ched>Q1<nl><fcel>North 1<fcel>21,132<nl></otsl>\n"
    "</doctag>"
)


def _fake_pipeline(calls: list | None = None) -> BlipVQAPipeline:
    def runner(image, question, max_new_tokens):
        if calls is not None:
            calls.append((image.mode, question, max_new_tokens))
        answer = " red " if "color" in question else "a small red house with a tree"
        return {"answer": answer, "new_tokens": 8}

    return BlipVQAPipeline(runner, "cpu", "float32", "injected")


def test_answer_output_fields_and_defaults():
    calls: list = []
    pipe = _fake_pipeline(calls)
    result = pipe.answer(Image.new("L", (400, 300)), "  what color is the   house? ")
    assert result["answer"] == "red"  # stripped
    assert result["question"] == "what color is the house?"  # whitespace collapsed
    assert result["image_size"] == [400, 300]
    assert result["new_tokens"] == 8 and result["truncated"] is False
    assert result["generation"] == {
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "do_sample": False,
        "decoding": "greedy",
    }
    assert (result["model_id"], result["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    assert (result["device"], result["dtype"], result["source"]) == ("cpu", "float32", "injected")
    assert calls == [("RGB", "what color is the house?", DEFAULT_MAX_NEW_TOKENS)]


def test_answer_reports_truncation():
    pipe = _fake_pipeline()
    result = pipe.answer(Image.new("RGB", (64, 64)), "what is in the picture?", max_new_tokens=8)
    assert result["truncated"] is True and result["answer"] == "a small red house with a tree"


def test_answer_rejects_bad_inputs():
    pipe = _fake_pipeline()
    with pytest.raises(TypeError):
        pipe.answer(np.zeros((30, 40, 3), dtype=np.uint8), "Who?")
    with pytest.raises(ValueError, match="MIN_IMAGE_SIDE"):
        pipe.answer(Image.new("RGB", (MIN_IMAGE_SIDE - 1, 64)), "Who?")
    with pytest.raises(ValueError, match="MAX_IMAGE_SIDE"):
        pipe.answer(Image.new("RGB", (MAX_IMAGE_SIDE + 1, 64)), "Who?")
    with pytest.raises(TypeError, match="question must be a str"):
        pipe.answer(Image.new("RGB", (64, 64)), None)
    with pytest.raises(ValueError, match="non-whitespace"):
        pipe.answer(Image.new("RGB", (64, 64)), "   ")
    with pytest.raises(ValueError, match="MAX_QUESTION_CHARS"):
        pipe.answer(Image.new("RGB", (64, 64)), "x" * (MAX_QUESTION_CHARS + 1))
    with pytest.raises(ValueError, match="MAX_NEW_TOKENS"):
        pipe.answer(Image.new("RGB", (64, 64)), "Who?", max_new_tokens=MAX_NEW_TOKENS + 1)
    with pytest.raises(TypeError, match="max_new_tokens"):
        pipe.answer(Image.new("RGB", (64, 64)), "Who?", max_new_tokens=True)


def test_answer_rejects_malformed_runner_output():
    pipe = BlipVQAPipeline(lambda *args: {"tokens": 1}, "cpu")
    with pytest.raises(RuntimeError, match="answer"):
        pipe.answer(Image.new("RGB", (64, 64)), "Who?")
