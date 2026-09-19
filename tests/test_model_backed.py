"""Model-backed checks that run only where the pinned snapshot is staged (local pre-flight): a referenced
evaluation, a one-epoch adaptation of the last decoder block on a dozen drawn scenes, the artifact round trip,
the loader's scope check, the transactional guarantee and — where CUDA is visible — the same path on the
accelerator. Skipped when the weights are absent."""

from __future__ import annotations

import hashlib
import json
import shutil

import pytest
import torch
from PIL import Image, ImageDraw

from blip_vqa_pipeline import DEFAULT_WEIGHTS_DIR, WEIGHT_FILE, BlipVQAPipeline

pytest.importorskip("transformers")
if not (DEFAULT_WEIGHTS_DIR / WEIGHT_FILE).is_file():
    pytest.skip("snapshot not staged", allow_module_level=True)

COLOURS = ["red", "green", "blue", "yellow"]


def _scene(path, colour, shape):
    image = Image.new("RGB", (256, 192), (135, 206, 235))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 130, 256, 192], fill=(60, 179, 75))
    if shape == "circle":
        draw.ellipse([80, 40, 176, 136], fill=colour)
    else:
        draw.rectangle([80, 40, 176, 136], fill=colour)
    image.save(path, format="PNG")
    return path


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    root = tmp_path_factory.mktemp("scenes")
    out = []
    for i in range(12):
        colour, shape = COLOURS[i % 4], "circle" if i % 2 else "square"
        path = _scene(root / f"scene{i}.png", colour, shape)
        out.append(
            {
                "id": f"s{i:02d}",
                "image_id": f"scene{i}",
                "image": str(path),
                "question": "what color is the shape?" if i % 3 else "is the sky blue?",
                "answers": [colour] * 8 + ["dark " + colour, "unanswerable"]
                if i % 3
                else ["yes"] * 9 + ["no"],
                "category": "other" if i % 3 else "yes/no",
            }
        )
    return out


def _answers(pipe, records):
    out = []
    for r in records:
        with Image.open(r["image"]) as image:
            image.load()
            out.append(pipe.answer(image, r["question"])["answer"])
    return out


@pytest.fixture(scope="module")
def pipe():
    return BlipVQAPipeline.from_pretrained(device="cpu")


def test_evaluate_scores_accepted_answers(pipe, records):
    metrics = pipe.evaluate(records[:5])
    assert metrics["n"] == 5 and 0.0 <= metrics["vqa_accuracy"] <= 1.0 and metrics["adapted"] is False
    assert set(metrics) >= {"exact_match", "anls", "majority_match", "empty_rate", "seconds"}


def test_one_epoch_adaptation_and_artifact_round_trip(pipe, records, tmp_path):
    result = pipe.adapt(records[:8], records[8:], epochs=1, trainable_decoder_layers=1, batch_size=4)
    assert result["n_trainable"] == 10_074_428  # last decoder block + head transform and bias
    assert result["history"][0]["note"] == "frozen model" and result["history"][1]["train_loss"] > 0.0
    assert all(
        name.startswith("text_decoder.bert.encoder.layer.11.")
        or name.startswith("text_decoder.cls.predictions.transform.")
        or name == "text_decoder.cls.predictions.bias"
        for name in result["trainable_names"]
    )
    assert (
        "text_decoder.cls.predictions.decoder.weight" not in result["trainable_names"]
    )  # tied to the embeddings
    artifact = pipe.save_artifact(tmp_path / "adapter", {"note": "test"})
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["tensors"]) == len(result["trainable_names"])
    reloaded = BlipVQAPipeline.from_artifact(artifact, device="cpu")
    assert _answers(pipe, records[:4]) == _answers(reloaded, records[:4])
    assert reloaded.adapter["best_epoch"] == result["best_epoch"]
    assert not any(p.requires_grad for p in pipe._model.parameters())


def test_no_validation_keeps_the_final_epoch_and_reloads_it(pipe, records, tmp_path):
    result = pipe.adapt(records[:8], None, epochs=2, trainable_decoder_layers=1, batch_size=4)
    assert result["best_epoch"] == 2 == result["epochs"] and result["selection"].startswith("final epoch")
    assert all(entry["val"] is None for entry in result["history"]) and len(result["history"]) == 3
    artifact = pipe.save_artifact(tmp_path / "final")
    reloaded = BlipVQAPipeline.from_artifact(artifact, device="cpu")
    state, other = pipe._model.state_dict(), reloaded._model.state_dict()
    assert all(torch.equal(state[name], other[name]) for name in result["trainable_names"])
    assert reloaded.adapter["best_epoch"] == 2 and reloaded.adapter["trainable_decoder_layers"] == 1


def test_load_artifact_refuses_a_tensor_set_that_differs_from_the_recorded_configuration(
    pipe, records, tmp_path
):
    from safetensors.torch import load_file, save_file

    pipe.adapt(records[:8], None, epochs=1, trainable_decoder_layers=1, batch_size=4)
    artifact = pipe.save_artifact(tmp_path / "ok")
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    fewer = tmp_path / "fewer"
    shutil.copytree(artifact, fewer)
    (fewer / "manifest.json").write_text(json.dumps({**manifest, "tensors": manifest["tensors"][:-1]}))
    with pytest.raises(ValueError, match="does not match its recorded configuration"):
        BlipVQAPipeline.from_artifact(fewer, device="cpu")
    extra = tmp_path / "extra"
    shutil.copytree(artifact, extra)
    tensors = load_file(str(extra / "adapter.safetensors"))
    tensors["zz.extra"] = torch.zeros(1)
    save_file(tensors, str(extra / "adapter.safetensors"), metadata={"format": "pt"})
    digest = hashlib.sha256((extra / "adapter.safetensors").read_bytes()).hexdigest()
    files = [
        {**manifest["files"][0], "bytes": (extra / "adapter.safetensors").stat().st_size, "sha256": digest}
    ]
    (extra / "manifest.json").write_text(json.dumps({**manifest, "files": files}))
    with pytest.raises(ValueError, match="tensor names differ"):
        BlipVQAPipeline.from_artifact(extra, device="cpu")
    other_layers = tmp_path / "other_layers"
    shutil.copytree(artifact, other_layers)
    adapter = {**manifest["adapter"], "trainable_decoder_layers": 2}
    (other_layers / "manifest.json").write_text(json.dumps({**manifest, "adapter": adapter}))
    with pytest.raises(ValueError, match="does not match its recorded configuration"):
        BlipVQAPipeline.from_artifact(other_layers, device="cpu")


def test_adapt_is_transactional_when_the_progress_callback_raises(pipe, records):
    before = {k: v.clone() for k, v in pipe._model.state_dict().items()}

    def boom(entry):
        if entry["epoch"] == 1:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        pipe.adapt(records[:8], None, epochs=2, trainable_decoder_layers=1, batch_size=4, progress=boom)
    after = pipe._model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before) and pipe.adapter is None
    assert not any(p.requires_grad for p in pipe._model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not visible")
def test_answer_adapt_and_reload_run_on_a_cuda_device(records, tmp_path):
    """Every tensor the runner and the trainer build must land on the model's device."""
    cuda = BlipVQAPipeline.from_pretrained(device="cuda:0")
    assert cuda.device == "cuda:0"
    with Image.open(records[0]["image"]) as image:
        image.load()
        first = cuda.answer(image, records[0]["question"])
    assert first["device"] == "cuda:0" and isinstance(first["answer"], str)
    result = cuda.adapt(records[:8], records[8:], epochs=1, trainable_decoder_layers=1, batch_size=4)
    assert result["best_epoch"] in (0, 1) and result["history"][1]["train_loss"] > 0.0
    artifact = cuda.save_artifact(tmp_path / "cuda")
    reloaded = BlipVQAPipeline.from_artifact(artifact, device="cuda:0")
    assert _answers(cuda, records[:4]) == _answers(reloaded, records[:4])
