"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "blip_vqa_pipeline",
    "repo_name": "blip-vqa-pipeline",
    "stem": "blip_vqa",
    "notebook_name": "blip_vqa_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "mode": "GUIDED",
    "pipeline_class": "BlipVQAPipeline",
    "weights_key": "blip-vqa-base",
    "runtime_imports": ["torch", "transformers"],
    "title": "BLIP VQA-base — DIMER visual question answering tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/blip-vqa-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/blip-vqa-pipeline/blob/main/tutorials/blip_vqa_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Salesforce%2Fblip--vqa--base-ffcc4d?style=flat",
            "https://huggingface.co/Salesforce/blip-vqa-base",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-salesforce%2FBLIP-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/salesforce/BLIP",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2201.12086-b31b1b.svg", "https://arxiv.org/abs/2201.12086"),
    ],
    "capability": "Visual question answering — one image plus one natural-language question → one short answer string — using the pinned `Salesforce/blip-vqa-base` weights",
    "intro": (
        "At inference the BLIP model (a ViT-B/16 image encoder at 384×384, a BERT-style question encoder that attends to the "
        "image features, and a 12-layer answer decoder; about 385M parameters, pretrained on 129M image–text pairs with "
        "captioning-and-filtering bootstrapping and fine-tuned on VQA v2) encodes the resized image, encodes the question "
        "against it, and generates the answer text token by token. Decoding is greedy (`do_sample=False`) under a "
        "caller-owned `max_new_tokens` budget. **No adaptation occurs:** no training, fine-tuning, in-context conditioning, "
        "or preprocessing fitting happens in this notebook — the upstream checkpoint supplies the weights, processor and "
        "tokenizer, and the carried module adds snapshot verification, the input contract (image side ceilings, a non-empty "
        "question up to 256 characters, the token budget), a fixed output contract, and the `exact_match`, `anls`, "
        "`vqa_accuracy`, `validate_inputs` and `evaluation_report` helpers. The default sample is a flat cartoon scene drawn "
        "in code with seven authored questions and accepted answers, so exact-match and ANLS are demonstration (plumbing) "
        "evidence for one drawing, not a VQA benchmark."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, resolve and digest-verify the "
        "immutable upstream model revision, draw a synthetic scene with authored question/answer pairs (or upload your own "
        "photograph and write your own questions) and validate it into an input manifest, choose a token budget, run the "
        "supported task, read the answers correctly (generated text, no score, a `truncated` flag), exercise an optional BYOD "
        "path, produce an evaluation report that is `sample-sanity` with `exact_match` and `anls` only when accepted answers "
        "exist and `not-measurable` otherwise, and export the answers, the annotated image and provenance."
    ),
    "exclusions": (
        "Reading text in the image (BLIP-VQA is not an OCR or document model; use a document-QA pipeline for that), "
        "counting beyond a few objects or spatial reasoning the model was not trained for, answer localisation (the model "
        "returns text, not a region), open-ended captioning (a separate checkpoint), batch throughput, sampling or beam "
        "search, evaluation on the VQA v2 benchmark (not bundled; only authored questions on a drawn scene are scored here), "
        "and any training. The model was fine-tuned on photographs with short English answers; drawings, diagrams, "
        "non-English questions and long free-text answers are outside what this notebook measures, and a fluent wrong "
        "answer carries no signal."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU and uses CUDA automatically when available; inference is float32 on both. CPU is adequate: the repository's model card records 5.8 s to load and about 0.2 s per question on the 640×480 drawn scene in the Windows venv (Intel Core Ultra 9 275HX). The pinned `torch==2.14.0` install and the 1.54 GB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python and PIL; what an encoder–decoder model's generated tokens are; what VQA accuracy (`min(matching human answers / 3, 1)`) and normalised Levenshtein similarity (ANLS) measure; that a confident answer is not a correct one.",
        "- **Data:** the default sample is a deterministic 640×480 cartoon scene drawn in code with Pillow (sky, grass, sun, a red house with a brown door, one tree, a white ball; no text rendering, so its digest is stable across Pillow builds) with seven authored questions and their accepted answers, so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one image decodable by Pillow (PNG/JPEG/WebP and similar), any colour mode, sides between 16 and 4096 px, plus your own questions typed into the form field. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Draw the synthetic scene or optional BYOD\n\n"
                "The default sample is **synthetic** and carries its own references: a flat cartoon scene — blue sky, green "
                "grass, a yellow sun, a red house with a brown roof and a brown door, one round tree and a white ball — is drawn "
                "with Pillow at 640×480, the same drawing the repository's smoke run used. Seven questions are authored against "
                "it, each with the accepted answer(s) as drawn; they are the references for the `exact_match` and `anls` sanity "
                "checks later. Two of them are **recorded misses** from the smoke run, kept on purpose: the model counted two "
                "trees where one is drawn and called the brown door `red`. They are not a labelled dataset, so nothing here is a "
                "VQA v2 measurement. The image digest is printed for the record. BYOD is optional and disabled by default; when "
                "enabled, upload one image and type your questions (one per line) — no accepted answers exist for them, so the "
                "evaluation report will be `not-measurable`.\n\n"
                "The token budget is a **caller-owned request parameter**: `max_new_tokens` bounds the answer "
                "(`DEFAULT_MAX_NEW_TOKENS = 10` fits any VQA-style answer; `MAX_NEW_TOKENS = 32` is the ceiling). Nothing "
                "is validated in this cell — the next section hands the image and the questions to the pipeline's own "
                "validation stage, which is the only checker. Look for a dictionary naming the sample kind, the image size and "
                "digest, the budget and the number of questions."
            ),
            "code": (
                "import hashlib\n"
                "import io\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw, ImageFont\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "byod_questions = 'what is in the picture?\\nwhat color is the largest object?'  # @param {{type:\"string\"}}\n"
                "max_new_tokens = 10  # @param {{type:\"integer\"}}\n\n\n"
                "def synthetic_scene(width=640, height=480):\n"
                "    \"\"\"A flat cartoon scene drawn with Pillow (no text); returns image + [(question, accepted answers)].\"\"\"\n"
                "    image = Image.new('RGB', (width, height), (135, 206, 235))  # sky\n"
                "    d = ImageDraw.Draw(image)\n"
                "    d.rectangle([0, 300, 640, 480], fill=(60, 179, 75))  # grass\n"
                "    d.ellipse([500, 40, 600, 140], fill=(255, 215, 0))  # sun\n"
                "    d.rectangle([120, 180, 320, 330], fill=(200, 40, 40))  # red house\n"
                "    d.polygon([(100, 180), (220, 90), (340, 180)], fill=(90, 50, 20))  # brown roof\n"
                "    d.rectangle([200, 260, 240, 330], fill=(70, 40, 20))  # brown door\n"
                "    d.ellipse([420, 260, 520, 360], fill=(40, 100, 40))  # tree crown\n"
                "    d.rectangle([460, 350, 480, 420], fill=(90, 60, 30))  # trunk\n"
                "    d.ellipse([60, 380, 140, 440], fill=(255, 255, 255))  # white ball\n"
                "    qa = [\n"
                "        ('what color is the house?', ['red']),\n"
                "        ('how many houses are there?', ['1', 'one']),\n"
                "        ('what is the yellow object?', ['sun', 'the sun']),\n"
                "        ('what is next to the house?', ['tree', 'a tree']),\n"
                "        ('what color is the sky?', ['blue']),\n"
                "        ('how many trees are there?', ['1', 'one']),  # smoke run answered 2: recorded miss\n"
                "        ('what color is the door?', ['brown', 'dark brown']),  # smoke run answered red: recorded miss\n"
                "    ]\n"
                "    return image, qa\n\n\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    image_name = next(iter(uploaded))\n"
                "    image = Image.open(io.BytesIO(uploaded[image_name]))\n"
                "    image.load()\n"
                "    questions = [line.strip() for line in byod_questions.splitlines() if line.strip()]\n"
                "    golds = None\n"
                "    sample_kind = 'BYOD'\n"
                "else:\n"
                "    # Deterministic drawing: no randomness and no text rendering, so no seed is needed and the digest is stable.\n"
                "    image, qa = synthetic_scene()\n"
                "    questions, golds = [q for q, _ in qa], [g for _, g in qa]\n"
                "    image_name = 'synthetic_scene_640x480.png'\n"
                "    sample_kind = 'synthetic'\n\n"
                "image_sha256 = hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()\n"
                "print({{'sample_kind': sample_kind, 'name': image_name, 'mode': image.mode, 'size': image.size, 'rgb_sha256': image_sha256, 'max_new_tokens': max_new_tokens, 'n_questions': len(questions), 'has_golds': golds is not None}})"
            ),
        },
        {
            "md": (
                "## 5. Validate the request → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `answer` applies — "
                "image type and sides `MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE` px, each question a non-empty string of at most "
                "`MAX_QUESTION_CHARS` characters (whitespace collapsed), and `max_new_tokens` in `[1, MAX_NEW_TOKENS]` — and "
                "returns an **input manifest** naming the schema (including the 384×384 resize that does not preserve aspect "
                "ratio, and the decoding rule), the input's observed mode and size, the checked questions, the budget and the "
                "verdict. The manifest is written to `outputs/{stem}_input_manifest.json`. To show what rejection looks like, "
                "the cell also validates a blank question and records the pipeline's own error message as a finding. Inside the "
                "pipeline the image is converted to RGB and resized to `IMAGE_SIZE`×`IMAGE_SIZE`; nothing else is dropped or "
                "altered. The pipeline cannot tell whether the question is answerable from the image: that contract is the "
                "caller's."
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'IMAGE_SIZE': IMAGE_SIZE, 'MAX_QUESTION_CHARS': MAX_QUESTION_CHARS, 'MAX_NEW_TOKENS': MAX_NEW_TOKENS, 'DEFAULT_MAX_NEW_TOKENS': DEFAULT_MAX_NEW_TOKENS, 'DECODING': DECODING}}}})\n"
                "input_manifest = validate_inputs(image, questions, max_new_tokens=max_new_tokens, names=[image_name])\n"
                "# Demonstrate rejection on a request that breaks the contract; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(image, ['   '])\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'blank-question-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        {
            "md": (
                "## 6. Answer the questions and read the output correctly\n\n"
                "`answer` returns, per question, a dict with `answer` (the decoded text, stripped), the checked `question`, "
                "`image_size`, `new_tokens`, a `truncated` flag that is true when the budget was exhausted, the generation "
                "settings and the model identity. **No score exists**: the answer is generated text with no probability and no "
                "correctness signal, and a fluent answer is not evidence that it describes the image. Greedy decoding is "
                "deterministic on a fixed device and dtype; CUDA kernel selection can change a token and therefore the rest of "
                "the answer, so GPU and CPU outputs need not match. Each call re-encodes the image with the question, so cost "
                "is per question (about 0.2 s each on the reference CPU). As recorded in the model card, the repository's CPU "
                "smoke on this same drawing answered five of the seven authored questions exactly, counted `2` trees and "
                "called the door `red` — and answered `dog` to \"what is in the picture?\" on a blank white image: the model "
                "always produces an answer, whether or not one exists."
            ),
            "code": (
                "import time\n\n"
                "results, seconds = [], []\n"
                "for question in questions:\n"
                "    t0 = time.time()\n"
                "    results.append(pipe.answer(image, question, max_new_tokens=max_new_tokens))\n"
                "    seconds.append(round(time.time() - t0, 2))\n"
                "print({{'device': pipe.device, 'dtype': pipe.dtype, 'seconds_per_question': seconds, 'any_truncated': any(r['truncated'] for r in results)}})\n"
                "for result in results:\n"
                "    print(f\"Q: {{result['question']}}\\n   A: {{result['answer']!r}}  ({{result['new_tokens']}} tokens{{', TRUNCATED' if result['truncated'] else ''}})\")\n"
                "if any(r['truncated'] for r in results):\n"
                "    print('A budget was exhausted: that answer is incomplete. Raise max_new_tokens (ceiling MAX_NEW_TOKENS) and rerun.')"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. No accuracy is "
                "reported by default: VQA accuracy needs labelled questions with several human answers each on images from "
                "the deployment domain, and this repository ships none (VQA v2 is not bundled). The repository's metric "
                "helpers are `exact_match` after normalisation (lower-case, punctuation removed, whitespace collapsed) against "
                "any accepted answer; `anls` — normalised Levenshtein similarity `1 − edits / max(len)`, maximised over the "
                "accepted answers, scored 0 below the 0.5 threshold — reported alongside for near-misses; and `vqa_accuracy`, "
                "the benchmark's `min(matching human answers / 3, 1)`, which the report does **not** use because with authored "
                "accepted answers it degenerates to exact match (one authored answer can never score above 1/3). When accepted "
                "answers are supplied the report carries the `exact_match` rate, the mean `anls` and one entry per question, "
                "with the verdict `sample-sanity`. On the synthetic path those answers are facts **you drew yourself**, so a "
                "high score proves only that the input contract, forward pass and decoding round-trip — and the two recorded "
                "misses show what a wrong answer looks like in the report. On BYOD no accepted answers exist, the verdict is "
                "`not-measurable`, and the report states what would make the task measurable. The report is written to "
                "`outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "report = evaluation_report(results, golds, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps({{k: v for k, v in report.items() if k not in ('metrics', 'per_question')}}, indent=2))\n"
                "for metric in report['metrics']:\n"
                "    print(f\"{{metric['id']:12}} {{metric['value']:.3f}}  ({{metric['estimation']}})\")\n"
                "for entry in report.get('per_question', []):\n"
                "    print(f\"  exact {{str(entry['exact_match']):5}}  anls {{entry['anls']:.2f}}  {{entry['question']}} -> {{entry['prediction']!r}} (accepted: {{entry['golds']}})\")\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No accepted answers exist for these questions, so nothing is scored; read the answers against the image yourself.')"
            ),
        },
        {
            "md": (
                "## 8. Export outputs and provenance\n\n"
                "Machine-readable JSON preserves every result (question, answer, `new_tokens`, `truncated`, the budget), the "
                "evaluation report, the input manifest, the sample identity, digest and accepted answers, the notebook's source "
                "(repository, revision, embedded module digest, generator), the model identifier, the immutable model revision, "
                "the model licence, and the runtime identity (Python, `torch`, `transformers`, device). The question/answer "
                "pairs are also written as CSV with explicit `image`, `question`, `answer`, `new_tokens`, `truncated` columns, "
                "and an annotated PNG shows the image with the questions and answers printed in a panel beneath it for visual "
                "inspection (the model returns no location, so nothing is drawn on the image itself) — a supplement to, not a "
                "replacement for, the machine-readable files. No credentials are recorded."
            ),
            "code": (
                "import csv\n\n"
                "panel_height = 30 + 26 * len(results)\n"
                "annotated = Image.new('RGB', (image.width, image.height + panel_height), 'white')\n"
                "annotated.paste(image.convert('RGB'), (0, 0))\n"
                "draw = ImageDraw.Draw(annotated)\n"
                "draw.line([(0, image.height + 1), (image.width, image.height + 1)], fill=(120, 120, 120), width=2)\n"
                "panel_font = ImageFont.load_default(size=16)\n"
                "for index, result in enumerate(results):\n"
                "    draw.text((20, image.height + 12 + 26 * index), f\"{{result['question']}}  ->  {{result['answer']}}\", fill=(40, 90, 220), font=panel_font)\n"
                "annotated.save('outputs/{stem}_annotated.png')\n"
                "payload = {{\n"
                "    'predictions': results,\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'sample': {{'kind': sample_kind, 'name': image_name, 'size': list(image.size), 'rgb_sha256': image_sha256, 'questions': questions, 'accepted_answers': golds}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'device': pipe.device,\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "with open('outputs/{stem}_answers.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['image', 'question', 'answer', 'new_tokens', 'truncated'])\n"
                "    for result in results:\n"
                "        writer.writerow([image_name, result['question'], result['answer'], result['new_tokens'], result['truncated']])\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The answers are the text the model generates for an image and a question; nothing in the output scores that text, "
        "the model returns no location or evidence, and it answers every question — including one about a blank image — with "
        "equal fluency. On the drawn scene the `exact_match` and `anls` values in the evaluation report compare the answers "
        "with facts you drew yourself and the verdict is `sample-sanity`, which proves only that the input contract, forward "
        "pass and decoding work (the repository's smoke run scored 5/7 exact on this drawing, miscounting the trees and "
        "miscolouring the door); they say nothing about photographs, cluttered scenes, counting, reading, spatial or "
        "commonsense reasoning, or answers longer than a phrase, and a BYOD result is a single-image observation with the "
        "verdict `not-measurable`. **The model answers any question about any image** and stops only at end-of-sequence or "
        "the token budget: check `truncated`, and treat a plausible answer to an unanswerable question as the expected failure "
        "mode, not an exception. The pipeline provides no OCR, no answer localisation, no captioning, no benchmark evaluation "
        "and no training capability.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, can "
        "acquire and digest-verify the pinned model, validate the demonstrated request, execute the public pipeline path, and "
        "emit the shown machine-readable outputs in the tested runtime — without the repository being reachable. It does **not** "
        "establish benchmark superiority, deployment calibration, safety for high-consequence decisions, or production fitness on "
        "an unseen domain.\n\n"
        "**Next experiments:** ask a question the drawing cannot answer (`what is the dog doing?`) and see the model invent "
        "one; ask `is it raining?` and `what time of day is it?`; lower `max_new_tokens` to 1 and watch `truncated` turn true "
        "on a two-token answer; enable `USE_BYOD` with a photograph you know, type your questions, then pass your own accepted "
        "answers to `evaluation_report` to see the verdict switch to `sample-sanity`.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/blip-vqa-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/blip-vqa-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/blip-vqa-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/salesforce/BLIP\n"
        "- BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation (Li et al., 2022): https://arxiv.org/abs/2201.12086\n"
        "- Making the V in VQA Matter — VQA v2 and the VQA accuracy metric (Goyal et al., 2017): https://arxiv.org/abs/1612.00837\n"
        "- Scene Text Visual Question Answering — the ANLS metric (Biten et al., 2019): https://arxiv.org/abs/1905.13648"
    ),
}
