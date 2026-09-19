"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, metrics.py, samples.py), and the model pin/stage/verify cells are produced by
the generator from repository sources so they cannot drift from the package.

This template configures an E2E visual-question-answering workflow: the pinned Salesforce/blip-vqa-base
snapshot is digest-verified and loaded, the annotations of a digest-pinned VizWiz-VQA shard are read
column-only and 240 pinned photographs taken by blind users are fetched one by one, validated and split
by image, seven authored questions over a drawn scene are answered through the inference contract, the
frozen model is scored on the held-out photographs beside two non-neural baselines, a bounded
fine-tuning of the answer decoder's last blocks runs in the kernel, the held-out split is scored again
per category, the adapted model re-answers the drawn scene, and the adapter is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "blip_vqa_pipeline",
    "repo_name": "blip-vqa-pipeline",
    "stem": "blip_vqa",
    "notebook_name": "blip_vqa_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
        "pinned `Salesforce/blip-vqa-base` snapshot (safetensors, 1.54 GB), reads the four text columns of one digest-pinned "
        "VizWiz-VQA shard from the Hugging Face Hub (about 0.13 MB over HTTP range requests, no credential), fetches the 240 "
        "pinned photographs one by one from the VizWiz project's image host (about 113 MB, each refused on any size or "
        "SHA-256 mismatch), cuts them by image into 140 / 40 / 60 training, validation and test questions, answers seven "
        "authored questions over a drawn scene through the inference contract with an input manifest and a rejection probe, "
        "scores the frozen model on the test photographs with VQA accuracy, exact match, majority match and ANLS beside the "
        "constant-answer and question-prefix baselines, runs a bounded fine-tuning of the answer decoder's last two blocks "
        "and head on the training photographs with validation-VQA-accuracy epoch selection, scores the held-out photographs "
        "again per category, re-answers the drawn scene with the adapted model, exports the adapter as safetensors with a "
        "manifest, and reloads that artifact into a fresh pipeline to verify answer parity. The default path needs no "
        "repository clone, no DIMER worker or service, no credential, no upload dialog and no configuration edit "
        "(NOTEBOOK_SPEC 2.0 §5). On CPU the whole path takes about ten minutes of model time after the downloads; a CUDA "
        "runtime is used automatically when present."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to upload one zip "
        "holding a `records.jsonl` (or `records.json`) of `{{id, image, question, answers}}` objects — `image` a file name inside "
        "the zip, `answers` one or more accepted answers — beside the image files. They pass through the same validation, "
        "seeded image-disjoint split, baselines, fine-tuning, held-out evaluation, artifact export and reload-parity cells as "
        "the VizWiz sample. The expected schema and the ceilings are stated in the Prerequisites and in Section 4, and uploaded "
        "files stay inside this runtime. BYOD is optional and never part of the default path."
    ),
    "pipeline_class": "BlipVQAPipeline",
    "weights_key": "blip-vqa-base",
    "modules": ["pipeline.py", "metrics.py", "samples.py"],
    "entry_module": "pipeline.py",
    "runtime_imports": ["torch", "transformers"],
    "title": "BLIP VQA-base — DIMER E2E visual question-answering fine-tuning tutorial (standalone)",
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
    "capability": "visual question answering and bounded supervised fine-tuning of the answer decoder's last blocks on a photograph/question/answers dataset, using the pinned `Salesforce/blip-vqa-base` weights",
    "intro": (
        "`Salesforce/blip-vqa-base` is the BLIP model of Li et al. (2022) — a ViT-B/16 image encoder at 384×384, a "
        "BERT-style question encoder that attends to the image features, and a 12-layer answer decoder; 361 M parameters, "
        "pretrained on 129 M image–text pairs and fine-tuned on VQA v2 and Visual Genome — published under the "
        "**BSD-3-Clause** licence. At inference it encodes the resized image, encodes the question against it, and generates "
        "the answer text token by token with greedy decoding under a caller-owned `max_new_tokens` budget. **No score "
        "exists**: the answer is generated text with no probability and no correctness signal, and a fluent answer is not "
        "evidence that it describes the image.\n\n"
        "What this notebook adds to inference is **adaptation with accepted answers**. The dataset is real and out of the "
        "model's distribution: VizWiz-VQA (Gurari et al., CVPR 2018; **CC BY 4.0**) — photographs taken by blind people with a "
        "spoken question each and ten crowd-sourced answers, about a quarter of them judged **unanswerable** (blur, framing, "
        "darkness) — a population the model never saw and a convention it does not know. The notebook reads only the four "
        "text columns of one pinned Hub shard (the shard holds its 864 images in one 406 MB row group; about 0.13 MB is "
        "fetched), then fetches the 240 photographs of the sample one by one from the VizWiz project's own host, each pinned "
        "by size and SHA-256. The frozen model answers VQA-v2-style — a short noun for every photograph, never "
        "`unanswerable` — so it scores below a constant answer on this population (the build record measured VQA accuracy "
        "0.24 frozen against 0.68 for the constant answer `unanswerable`), and the honest question is twofold: does a "
        "bounded adaptation of the answer decoder's last two blocks on 140 photographs learn the unanswerable convention "
        "*and* improve the answerable questions, read **per category**? Four metrics are implemented in the carried modules "
        "(**VQA accuracy**, the benchmark's `min(matching answers / 3, 1)`; **exact match** against any accepted answer; "
        "**majority match** against the most frequent one, which a constant answer cannot game; and **ANLS**), and two "
        "**non-neural baselines** — the constant answer and the question-prefix majority — show where a system with no "
        "image sits. Nothing here is a quality claim about your photographs: it is one seeded split of one corpus.\n\n"
        "**Snapshot note:** the pinned revision ships a fast `tokenizer.json` and a float32 `model.safetensors` (an 8-file "
        "manifest) — no pickle is opened anywhere in this notebook. Section 3 stages and digest-verifies those files before "
        "the processor or the model is constructed."
    ),
    "learning_objectives": (
        "install the pinned runtime; read what the carried pipeline, metrics and dataset modules guarantee; stage and "
        "digest-verify the immutable upstream snapshot; fetch the annotations of a digest-pinned VQA corpus without "
        "downloading its shard and its photographs one by one with per-file digests, validate them and split by image without "
        "leakage; answer through the public API over a drawn scene and read `answer`, `new_tokens` and `truncated` correctly "
        "(generated text, no score); score the frozen model against ten accepted answers per question beside two non-neural "
        "baselines and read the per-category breakdown; run a bounded fine-tuning with explicit hyperparameters and "
        "validation-based epoch selection; evaluate on an image-disjoint test split; re-answer a drawing from a different "
        "image family with the adapted model; and export a safetensors adapter that reloads against the pinned base with "
        "verified parity."
    ),
    "exclusions": (
        "reading text in the image (BLIP-VQA is not an OCR or document model), counting beyond a few objects or spatial "
        "reasoning, answer localisation (the model returns text, not a region), open-ended captioning (a separate checkpoint), "
        "batch throughput, sampling or beam search, evaluation on the VQA v2 or VizWiz benchmarks proper (only one seeded "
        "240-question sample is scored here), fine-tuning of the vision encoder, the question encoder, the embeddings or the "
        "tied output projection, training on images that are not the pinned sample or your own uploads, and any claim that a "
        "VizWiz split stands in for your photographs. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available. CPU is adequate but not fast: the build record measured about 7 s to load and digest-verify the 1.54 GB snapshot, about 57 s to answer the 60 test photographs, and about 340 s for the four epochs of fine-tuning the answer decoder's last two blocks and head, including the one-off encoding of the 140 training photographs and the per-epoch validation scoring. The pinned `torch==2.14.0` install and the 1.54 GB checkpoint are the large downloads of the run; the 240 photographs add about 113 MB.",
        "- **Knowledge:** basic Python and PIL; what an encoder–decoder model's generated tokens are; what VQA accuracy (`min(matching human answers / 3, 1)`), exact match, majority match and ANLS measure and why none is a human judgement; why a confident answer is not a correct one.",
        "- **Data contract:** records are `{{id, image, question, answers}}` — an image file decodable by Pillow with sides between `MIN_IMAGE_SIDE` (16) and `MAX_IMAGE_SIDE` (4096) px, a question of at most `MAX_QUESTION_CHARS` (256) characters, and one or more non-empty accepted answers (`MIN_ANSWERS` = 1; VizWiz supplies ten); optional `image_id` (defaults to the id) groups questions on the same image and optional `category` labels the breakdown. Ids match `[A-Za-z0-9_.:-]{{1,64}}` and are unique; a dataset needs 8..5,000 records; every question on the same image lands in the same split so a test image is never trained on; the training target is each record's majority answer. BYOD accepts one zip of images plus a `records.jsonl` / `records.json` in that shape.",
        "- **Validation is structural, not semantic:** every image is opened and decoded and every question checked as `answer` checks it, but nothing checks that an accepted answer is right — a mislabelled corpus is fine-tuned on without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — photographs of people, documents or homes with their questions are exactly that. The default path uploads nothing.",
        "- **External access (data):** besides the model snapshot, the default path reads the four text columns of one object in the Hub dataset repository `lmms-lab-encoder/VizWiz-VQA` at the immutable revision `d428a2da…` (`data/val-00000-of-00005-…parquet`, 405,819,917 bytes, SHA-256 `62a5fddb…`): the declared size and SHA-256 are checked against the pins before any byte is read, only the parquet footer and those columns are fetched over HTTPS range requests, and the decoded columns are refused unless their SHA-256 matches; it then fetches 240 JPEG files from `vizwiz.cs.colorado.edu` (about 113 MB in total, each pinned by size and SHA-256 in the carried module and refused on any mismatch, with a short pause between requests and a bounded retry on transient errors). The corpus is CC BY 4.0 (Gurari et al., 2018).",
    ],
    "cells": [
        {
            "md": (
                "## 4. VizWiz photographs, annotations and split\n\n"
                "`fetch_annotations` reads the pinned shard's four text columns (or the cache under `weights/vizwiz/`): it "
                "first checks the byte size and SHA-256 the Hub declares for the file against the pins, then reads only the "
                "parquet footer and those column chunks through `pyarrow` over HTTPS range requests, and refuses the decoded "
                "columns unless their SHA-256 matches. `fetch_images` stages the 240 pinned photographs from the VizWiz host "
                "one by one — each cached file is re-hashed, each fetched file refused on any size or SHA-256 mismatch, with "
                "a short pause between requests and a bounded retry on transient errors. `build_sample_dataset` takes exactly "
                "the pinned questions, shuffles them with `SPLIT_SEED` and cuts them **by image** into 140 / 40 / 60 training, "
                "validation and test records (VizWiz has one question per photograph). `validate_dataset` then opens and "
                "decodes every image and checks every record against the contract, `check_split_disjoint` asserts no image "
                "is shared, and the training split is written to `outputs/{stem}_train.jsonl` in the shape BYOD expects.\n\n"
                "Look for: 864 annotation rows, 240 photographs, three digests, the category mix per split (about half the "
                "sample is `unanswerable`, a larger share than the benchmark's quarter because the shard's own mix is what "
                "the seeded draw sampled from), and four refusal probes — a duplicate id, a missing image file, an empty "
                "answer list and a dataset too small to split — each rejected before `torch` does anything."
            ),
            "code": (
                "import collections\n"
                "import hashlib\n"
                "import io\n"
                "import json\n"
                "import zipfile\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_dir = Path('work') / 'byod'\n"
                "    byod_dir.mkdir(parents=True, exist_ok=True)\n"
                "    with zipfile.ZipFile(io.BytesIO(payload)) as archive:\n"
                "        for member in archive.infolist():\n"
                "            name = Path(member.filename).name\n"
                "            if member.is_dir() or not name or name.startswith('.'):\n"
                "                continue\n"
                "            (byod_dir / name).write_bytes(archive.read(member))\n"
                "    records_file = next(p for p in (byod_dir / 'records.jsonl', byod_dir / 'records.json') if p.is_file())\n"
                "    records = load_byod_dataset(records_file)\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED, base_dir=byod_dir)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_rows = {{'byod': len(records)}}\n"
                "else:\n"
                "    annotations = fetch_annotations(cache_dir='weights/vizwiz')\n"
                "    image_paths = fetch_images(sorted(IMAGE_PINS), cache_dir='weights/vizwiz')\n"
                "    raw_rows = {{'annotations': len(annotations), 'photographs': len(image_paths)}}\n"
                "    splits = build_sample_dataset(annotations, seed=SPLIT_SEED, image_paths=image_paths)\n"
                "    data_source = f'{{CORPUS_NAME}} {{CORPUS_RELEASE}} ({{CORPUS_LICENSE}})'\n"
                "dataset_manifests = {{name: validate_dataset(part) for name, part in splits.items()}}\n"
                "splits = {{name: manifest['records'] for name, manifest in dataset_manifests.items()}}\n"
                "disjoint = check_split_disjoint(splits)\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "categories = {{name: manifest['categories'] for name, manifest in dataset_manifests.items()}}\n"
                "write_dataset_jsonl(splits['train'], 'outputs/{stem}_train.jsonl')\n"
                "print({{'data_source': data_source, 'raw_rows': raw_rows, 'splits': disjoint, 'column_sha256': CORPUS_FILE['column_sha256'][:16] + '...', 'pinned_photographs': len(IMAGE_PINS)}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'unique_images': manifest['unique_images'], 'categories': manifest['categories'], 'answers_per_question': manifest['answers_per_question'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "example = splits['train'][0]\n"
                "print({{'example': {{'id': example['id'], 'image': Path(example['image']).name, 'size': example['image_size'], 'question': example['question'], 'answers': example['answers'], 'category': example['category']}}}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in splits['train'][:8]],\n"
                "    'missing image file': [{{**splits['train'][0], 'image': 'work/does-not-exist.jpg'}}, *splits['train'][1:8]],\n"
                "    'empty answer list': [{{**splits['train'][0], 'answers': []}}, *splits['train'][1:8]],\n"
                "    'too small': splits['train'][:3],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "## 5. Answer through the inference contract\n\n"
                "The inference contract is exercised as the inference-only tutorial exercised it: a flat cartoon scene is "
                "drawn in code with Pillow (no text rendering, so its digest is stable across builds) with seven authored "
                "questions and accepted answers — a different image family from the photographs, and a scene the model will "
                "be asked about again after adaptation. `validate_inputs` applies exactly the checks `answer` applies (image "
                "sides `MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE`, questions up to `MAX_QUESTION_CHARS`, `max_new_tokens` in "
                "`[1, MAX_NEW_TOKENS]`) and returns an input manifest; a blank question is validated too and its rejection "
                "recorded as a finding. `answer` returns the decoded text, the checked question, `image_size`, `new_tokens`, a "
                "`truncated` flag and the model identity. **No score exists**: the answer is generated text with no "
                "probability and no correctness signal — a fluent answer is **not evidence that it describes the image**, and "
                "greedy decoding is a reproducibility property, not a quality one. As recorded in the model card, the "
                "repository's CPU smoke on this same drawing answered five of the seven questions exactly, counted `2` trees "
                "and called the door `red`. Whether the answers are *right* is what Section 6 measures on 60 photographs "
                "with ten human answers each, not what seven authored pairs can tell you."
            ),
            "code": (
                "import time\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw\n\n"
                "ANSWER_MAX_TOKENS = 10  # @param {{type:\"integer\"}}\n\n\n"
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
                "image, qa = synthetic_scene()\n"
                "questions, golds = [q for q, _ in qa], [g for _, g in qa]\n"
                "image_name = 'synthetic_scene_640x480.png'\n"
                "image_sha256 = hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()\n"
                "ceilings = {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'IMAGE_SIZE': IMAGE_SIZE, 'MAX_QUESTION_CHARS': MAX_QUESTION_CHARS, 'MAX_NEW_TOKENS': MAX_NEW_TOKENS, 'DEFAULT_MAX_NEW_TOKENS': DEFAULT_MAX_NEW_TOKENS, 'DECODING': DECODING, 'MIN_RECORDS': MIN_RECORDS, 'MAX_RECORDS': MAX_RECORDS, 'MIN_ANSWERS': MIN_ANSWERS}}\n"
                "print(ceilings)\n"
                "input_manifest = validate_inputs(image, questions, max_new_tokens=ANSWER_MAX_TOKENS, names=[image_name])\n"
                "try:\n"
                "    validate_inputs(image, ['   '])\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'blank-question-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print({{'scene': image_name, 'size': image.size, 'rgb_sha256': image_sha256[:16] + '...', 'manifest_verdict': input_manifest['verdict'], 'findings': len(input_manifest['findings'])}})\n"
                "results = []\n"
                "for question in questions:\n"
                "    started = time.perf_counter()\n"
                "    result = pipe.answer(image, question, max_new_tokens=ANSWER_MAX_TOKENS)\n"
                "    results.append({{'seconds': round(time.perf_counter() - started, 3), **result}})\n"
                "    print(f\"Q: {{result['question']}}\\n   A: {{result['answer']!r}}  ({{result['new_tokens']}} tokens{{', TRUNCATED' if result['truncated'] else ''}})\")\n"
                "checks = {{\n"
                "    'one_result_per_question': len(results) == len(questions),\n"
                "    'answers_are_text': all(isinstance(r['answer'], str) for r in results),\n"
                "    'budget_respected': all(r['new_tokens'] <= ANSWER_MAX_TOKENS for r in results),\n"
                "    'setting_echoed': all(r['generation']['max_new_tokens'] == ANSWER_MAX_TOKENS and r['generation']['do_sample'] is False for r in results),\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'answer output failed a sanity check: {{checks}}')\n"
                "frozen_scene = evaluation_report(results, golds, sample_kind='synthetic')\n"
                "print({{'checks': checks, 'frozen_scene': {{m['id']: round(m['value'], 3) for m in frozen_scene['metrics']}}, 'verdict': frozen_scene['verdict'], 'any_truncated': any(r['truncated'] for r in results)}})"
            ),
        },
        {
            "md": (
                "## 6. Baselines and the frozen model's score on the test photographs\n\n"
                "Three numbers frame the adaptation, each read four ways. The **constant-answer baseline** replies "
                "`unanswerable` to everything: on VizWiz that word sits among the ten answers of most questions, so it scores "
                "high on VQA accuracy and exact match — which is exactly why **majority match** (the prediction equals the "
                "question's most frequent answer) is reported beside them. The **question-prefix baseline** replies with the "
                "training majority answer for the question's first two words (`what color` → the commonest colour) — a "
                "lookup with no model and no image. The **frozen model** answers the 60 test photographs with the "
                "`max_new_tokens` from Section 5 and is scored with the same four metrics: **VQA accuracy** "
                "(`min(matching human answers / 3, 1)`), **exact match** against any accepted answer, **majority match** and "
                "**ANLS**, all after lower-casing, punctuation removal and whitespace collapsing. Expect the frozen model to "
                "lose to the constant answer on the whole set — it answers every photograph with a VQA-v2-style noun and "
                "never says `unanswerable` — and read the per-category breakdown: the build record measured VQA accuracy "
                "0.23 on the answerable `other` questions and 0.08 on the `unanswerable` ones."
            ),
            "code": (
                "baseline_constant = constant_answer_baseline(test_records)\n"
                "baseline_prefix = question_prefix_baseline(train_records, test_records)\n"
                "METRICS = ('vqa_accuracy', 'exact_match', 'majority_match', 'anls')\n"
                "print({{'constant_answer_baseline': {{k: round(baseline_constant[k], 3) for k in METRICS}}, 'n': baseline_constant['n']}})\n"
                "print({{'question_prefix_baseline': {{k: round(baseline_prefix[k], 3) for k in METRICS}}, 'note': baseline_prefix['baseline']}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records, max_new_tokens=ANSWER_MAX_TOKENS)\n"
                "print({{'frozen_model_test': {{k: round(frozen_test[k], 3) for k in METRICS}}, 'n': frozen_test['n'], 'verdict': frozen_test['verdict'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n\n\n"
                "def by_category(predict, records):\n"
                "    scores = collections.defaultdict(list)\n"
                "    for record in records:\n"
                "        scores[record.get('category', 'byod')].append(vqa_accuracy(predict(record), gold_texts(record)))\n"
                "    return {{category: {{'n': len(values), 'vqa_accuracy': round(sum(values) / len(values), 2)}} for category, values in sorted(scores.items())}}\n\n\n"
                "def model_answer(pipeline):\n"
                "    def predict(record):\n"
                "        with Image.open(record['image']) as photo:\n"
                "            photo.load()\n"
                "            return pipeline.answer(photo, record['question'], max_new_tokens=ANSWER_MAX_TOKENS)['answer']\n"
                "    return predict\n\n\n"
                "prefix_table = question_prefix_answers(train_records)\n"
                "constant_fields = by_category(lambda record: 'unanswerable', test_records)\n"
                "prefix_fields = by_category(lambda record: prefix_table.get(question_prefix(record['question']), prefix_table['']), test_records)\n"
                "frozen_fields = by_category(model_answer(pipe), test_records)\n"
                "print({{'by_category': {{'constant': constant_fields, 'prefix': prefix_fields, 'frozen': frozen_fields}}}})\n"
                "for record in test_records[:3]:\n"
                "    print({{'category': record['category'], 'question': record['question'][:70], 'frozen': model_answer(pipe)(record), 'gold': gold_texts(record)[:4]}})\n"
                "assert frozen_test['vqa_accuracy'] > 0.0"
            ),
        },
        {
            "md": (
                "## 7. Bounded fine-tuning of the answer decoder's last blocks\n\n"
                "`pipe.adapt` trains only the last `TRAINABLE_DECODER_LAYERS` blocks of the answer decoder plus its output "
                "head's transform and bias — two blocks by default, 19,526,204 of 361,230,140 parameters; the vision encoder, "
                "the question encoder, every embedding and the head's projection weight (tied to the word embeddings) stay "
                "frozen. The frozen vision encoder's output is computed **once** per training photograph and reused across "
                "epochs, so an epoch costs a text-encoder and decoder pass per record. The target is each record's majority "
                "answer, decoded from the `[DEC]` start token with the model's own label-smoothed cross-entropy; AdamW at a "
                "fixed learning rate, gradient clipping at 1.0, seeded shuffling and no scheduler. Epoch 0 records the frozen "
                "model's validation VQA accuracy; every epoch is scored on the 40 validation photographs, and the epoch with "
                "the highest validation VQA accuracy is kept — forty questions make that selection noisy, which is why the "
                "held-out split in Section 8 is what the numbers are read from.\n\n"
                "Watch validation VQA accuracy jump in the first epoch (mostly the unanswerable convention) and then move "
                "slowly (the answerable questions). The build record's counter-examples are in the model card; the default is "
                "the smallest configuration that captured most of the gain."
            ),
            "code": (
                "EPOCHS = 4  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 5e-5  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 8  # @param {{type:\"integer\"}}\n"
                "TRAINABLE_DECODER_LAYERS = 2  # @param {{type:\"integer\"}}\n\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row.update({{'val_' + k: round(entry['val'][k], 3) for k in METRICS}})\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable_decoder_layers=TRAINABLE_DECODER_LAYERS, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test photographs were never used for training or epoch selection, and no test image appears in the "
                "training or validation splits. The adapted model is scored exactly as the frozen model was in Section 6, the "
                "four systems are put side by side on all four metrics, and the per-category breakdown is repeated. Read it "
                "in this order: **majority match** first (the metric a constant answer cannot game), then VQA accuracy per "
                "category — the `unanswerable` questions, where the adapted model should now match the constant answer, and "
                "the answerable `other` questions, where the build record measured 0.23 frozen → about 0.57 adapted while the "
                "constant answer scores what its partial credit allows. The cell asserts the adapted VQA accuracy is above the "
                "frozen one. Sixty photographs from one seeded split of one corpus give **no dispersion estimate**; the deltas "
                "are sample-sanity evidence that the adaptation contract works, not a benchmark, and a gain on VizWiz says "
                "nothing about your photographs until you measure it there."
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records, max_new_tokens=ANSWER_MAX_TOKENS)\n"
                "adapted_val = pipe.evaluate(val_records, max_new_tokens=ANSWER_MAX_TOKENS)\n"
                "adapted_fields = by_category(model_answer(pipe), test_records)\n"
                "comparison = {{metric: {{'constant': round(baseline_constant[metric], 3), 'prefix': round(baseline_prefix[metric], 3), 'frozen': round(frozen_test[metric], 3), 'adapted': round(adapted_test[metric], 3)}} for metric in METRICS}}\n"
                "comparison['delta_vs_frozen'] = {{metric: round(adapted_test[metric] - frozen_test[metric], 3) for metric in METRICS}}\n"
                "comparison['by_category'] = {{category: {{'n': frozen_fields[category]['n'], 'constant': constant_fields[category]['vqa_accuracy'], 'prefix': prefix_fields[category]['vqa_accuracy'], 'frozen': frozen_fields[category]['vqa_accuracy'], 'adapted': adapted_fields[category]['vqa_accuracy']}} for category in frozen_fields}}\n"
                "for key, row in comparison.items():\n"
                "    print({{key: row}})\n"
                "for record in test_records[:3]:\n"
                "    print({{'category': record['category'], 'question': record['question'][:70], 'adapted': model_answer(pipe)(record), 'gold': gold_texts(record)[:4]}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'categories': categories,\n"
                "    'max_new_tokens': ANSWER_MAX_TOKENS,\n"
                "    'baselines': {{'constant_answer': baseline_constant, 'question_prefix': baseline_prefix}},\n"
                "    'frozen_test': frozen_test,\n"
                "    'validation_metrics': adapted_val,\n"
                "    'test_metrics': adapted_test,\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "assert adapted_test['vqa_accuracy'] > frozen_test['vqa_accuracy']\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "## 9. Re-answer the drawn scene, export the adapter and reload it\n\n"
                "The seven authored questions from Section 5 are answered again by the adapted model — a drawing, a different "
                "image family from the photographs it was tuned on, so this is a small look at what the adaptation did "
                "*outside* its corpus (the build record's answers are in the model card; a changed answer here is a finding to "
                "record, not a failure) — and scored with the per-image `evaluation_report`, whose verdict is `sample-sanity` "
                "because seven authored pairs on one drawing carry no dispersion estimate. Both answer sets are written as "
                "CSV.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the answer decoder's last two blocks and its head transform "
                "and bias, about 78 MB — as `adapter.safetensors`, with a `manifest.json` recording the artifact format, the "
                "base model id and revision, the digest of the base `model.safetensors`, the tensor names, the file size and "
                "SHA-256, the training configuration and the epoch history (OUT8). `BlipVQAPipeline.from_artifact` re-verifies "
                "the base snapshot, checks the artifact manifest, its digest and its exact tensor set **before** deserialising, "
                "refuses any tensor that is not an answer-decoder tensor, and overlays the tensors onto a freshly loaded base — "
                "a new object from files, not the in-memory model (VER2). The cell asserts identical answers on eight test "
                "photographs (VER4)."
            ),
            "code": (
                "import csv\n"
                "import shutil\n\n"
                "adapted_results = [pipe.answer(image, question, max_new_tokens=ANSWER_MAX_TOKENS) for question in questions]\n"
                "adapted_scene = evaluation_report(adapted_results, golds, sample_kind='synthetic')\n"
                "for before, after, gold in zip(results, adapted_results, golds, strict=True):\n"
                "    print({{'question': before['question'], 'frozen': before['answer'], 'adapted': after['answer'], 'gold': gold}})\n"
                "print({{'scene_after_adaptation': {{m['id']: round(m['value'], 3) for m in adapted_scene['metrics']}}, 'verdict': adapted_scene['verdict']}})\n"
                "with open('outputs/{stem}_answers.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['image', 'question', 'frozen_answer', 'adapted_answer', 'adapted_new_tokens', 'gold'])\n"
                "    for before, after, gold in zip(results, adapted_results, golds, strict=True):\n"
                "        writer.writerow([image_name, after['question'], before['answer'], after['answer'], after['new_tokens'], ' | '.join(gold)])\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = BlipVQAPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "before = [model_answer(pipe)(r) for r in test_records[:8]]\n"
                "after = [model_answer(reloaded)(r) for r in test_records[:8]]\n"
                "parity = {{'identical_answers': sum(a == b for a, b in zip(before, after, strict=True)), 'of': len(before)}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['identical_answers'] == parity['of']\n\n"
                "weight_entry = next(entry for entry in MANIFEST['files'] if entry['path'] == WEIGHT_FILE)\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': snapshot['files'], 'total_bytes': snapshot.get('total_bytes'), 'fetched_this_run': fetched, 'weight_file': WEIGHT_FILE, 'weight_format': 'safetensors, digest-verified', 'weight_sha256': weight_entry['sha256']}},\n"
                "    'data_source': data_source,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'repo': CORPUS_REPO, 'revision': CORPUS_REVISION, 'release': CORPUS_RELEASE, 'license': CORPUS_LICENSE, 'columns': list(CORPUS_COLUMNS), 'file': CORPUS_FILE, 'image_host': IMAGE_HOST, 'pinned_images': len(IMAGE_PINS)}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'sanity_checks': checks, 'scene': {{'name': image_name, 'size': list(image.size), 'rgb_sha256': image_sha256}}, 'items': [{{k: r[k] for k in ('question', 'answer', 'new_tokens', 'truncated', 'seconds')}} for r in results], 'golds': golds, 'frozen_report': frozen_scene, 'adapted_report': adapted_scene}},\n"
                "    'comparison': comparison,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': pipe.device, 'dtype': 'float32', 'source': pipe.source}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The frozen model answers every photograph with a VQA-v2-style noun and never says `unanswerable`, so on VizWiz it "
        "scores below a constant answer on VQA accuracy, and a bounded fine-tuning of the answer decoder's last two blocks "
        "and head on 140 photographs does two things the breakdown separates: it learns the unanswerable convention (the "
        "`unanswerable` category goes from near zero to near the constant answer) and it lifts the answerable `other` "
        "questions well above both the frozen model and the baselines, with a 78 MB adapter that reloads to identical "
        "answers. That is the claim: the adaptation contract works end to end on a real out-of-distribution VQA corpus, and "
        "the numbers it produces are read per category against two non-neural baselines and the frozen model rather than "
        "in isolation.\n\n"
        "The test split is 60 photographs from one seeded draw of one shard of one corpus, the validation split that picks "
        "the epoch is 40, the metrics are four reference-based scores (own implementations of the VQA accuracy convention "
        "and DocVQA's normalisation, none a human judgement), and VizWiz's ten answers disagree often enough that "
        "`unanswerable` sits among the accepted answers of most questions — which is why majority match is reported and why "
        "the constant baseline looks strong. So a gain here says the contract works, not that the adapted model is better on "
        "your photographs, that it reads text, counts, or localises, or that its answers are faithful — it still generates "
        "an answer for every image, now including `unanswerable` for images it cannot read, and it can say so wrongly. "
        "Fine-tuning on a narrow corpus can also erode the model elsewhere; the drawn scene re-answered in Section 9 is one "
        "image of evidence about that, not a measurement.\n\n"
        "Three things to carry to real data. **Baselines first:** the constant-answer and question-prefix baselines and the "
        "frozen model's score on *your* accepted answers are the numbers to read before any adapted one, per category and "
        "on majority match. **Leakage:** keep every question on an image in one split (the contract does this) and split by "
        "photographer or session when your images come from few sources, never at random over near-duplicate frames. "
        "**Targets:** the training target is the majority answer; a corpus whose annotators disagree trains the model on "
        "whichever answer won the vote, and the other nine are only used for scoring.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can acquire and digest-verify the pinned model snapshot, fetch and digest-verify the annotations and "
        "photographs of a real VQA corpus, validate the demonstrated dataset contract without leakage, execute the inference "
        "contract and a bounded fine-tuning, evaluate against two trivial baselines and the frozen model on an image-disjoint "
        "split, and emit the shown machine-readable artifacts — without the repository being reachable. It does **not** "
        "establish benchmark superiority, VQA accuracy on any other population or camera, a usable answerability threshold, "
        "or production fitness.\n\n"
        "**Optional experiments (they do not affect the default path):** set `TRAINABLE_DECODER_LAYERS = 4` and compare the "
        "artifact size and the per-category scores; raise `EPOCHS` and watch the validation VQA accuracy pick the epoch; ask "
        "the adapted model `what is written on the sign?` about a photograph with text and read a confident wrong answer; or "
        "bring your own photographs through BYOD and read the two baselines before the adapted number.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/blip-vqa-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/blip-vqa-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/blip-vqa-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model (Salesforce, BSD-3-Clause): https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/salesforce/BLIP\n"
        "- BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation (Li et al., 2022): https://arxiv.org/abs/2201.12086\n"
        "- VizWiz Grand Challenge: Answering Visual Questions from Blind People (Gurari et al., CVPR 2018; VizWiz-VQA, CC BY 4.0): https://arxiv.org/abs/1802.08218 — data: https://vizwiz.org/tasks-and-datasets/vqa/\n"
        "- VQA v2 and the VQA accuracy metric (Goyal et al., CVPR 2017): https://arxiv.org/abs/1612.00837\n"
        "- Scene Text Visual Question Answering — the ANLS metric (Biten et al., 2019): https://arxiv.org/abs/1905.13648\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
