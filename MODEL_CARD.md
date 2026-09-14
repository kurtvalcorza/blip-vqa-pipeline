---
license: bsd-3-clause
model_card_spec: "1.1"
pipeline_tag: visual-question-answering
base_model: Salesforce/blip-vqa-base
date_published: "2022-12-12"
date_published_source: "Hugging Face Hub repository creation date of the exact hosted checkpoint (`createdAt` 2022-12-12T17:51:53Z, https://huggingface.co/api/models/Salesforce/blip-vqa-base — the Transformers-format conversion); the BLIP paper and original checkpoints are from 2022-01 (arXiv:2201.12086), and the pinned revision is the Hub's `main` as of 2026-09-14"
---

# BLIP VQA-base — Visual Question Answering (Inference)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Salesforce%2Fblip--vqa--base-ffcc4d?style=flat)](https://huggingface.co/Salesforce/blip-vqa-base)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-salesforce%2FBLIP-181717?style=flat&logo=github&logoColor=white)](https://github.com/salesforce/BLIP)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2201.12086-b31b1b.svg)](https://arxiv.org/abs/2201.12086)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This pipeline provides a ready-to-run interactive Google Colab notebook that exercises the repository's public API end to end — stage and verify the pinned upstream revision in a fresh runtime, validate an input, run the task, and inspect and export the outputs:

- **Task Inference Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/blip-vqa-pipeline/blob/main/tutorials/blip_vqa_colab.ipynb) [`blip_vqa_colab.ipynb`](https://github.com/kurtvalcorza/blip-vqa-pipeline/blob/main/tutorials/blip_vqa_colab.ipynb)  
  *Seven authored questions over a cartoon scene drawn in code with the pinned `Salesforce/blip-vqa-base` weights: one answer string per question under a caller-owned token budget, and `exact_match` / `anls` against the authored answers as sanity evidence only (two recorded misses kept on purpose) — no VQA v2 benchmark.*

---

#### Description

`Salesforce/blip-vqa-base` is the Transformers-format release of the BLIP *base* model fine-tuned for visual question answering, from "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation" (Li, Li, Xiong, Hoi, arXiv:2201.12086), converted by the Hugging Face team and pinned here to revision `787b3d35d57e49572baabd22884b3d5a05acf072` (the Hub's `main` on 2026-09-14). The snapshot `config.json` declares `BlipForQuestionAnswering`: a 12-layer ViT-B/16 image encoder at 384×384 (hidden size 768, 12 heads, 576 patches plus a class token), a 12-layer BERT-style text encoder (hidden size 768, vocabulary 30,524, maximum 512 positions) whose layers cross-attend to the image features, and a 12-layer text decoder of the same shape that generates the answer — about 385M parameters in the 1.54 GB float32 `model.safetensors`. The model was pretrained on 129M image–text pairs (COCO, Visual Genome, Conceptual Captions 3M/12M, SBU and a LAION subset, bootstrapped with BLIP's captioning-and-filtering procedure) and fine-tuned on VQA v2 (with Visual Genome QA, per the paper) as an answer-generation task. At inference the processor (`BlipImageProcessor`, `preprocessor_config.json`: resize to 384×384 without preserving aspect ratio, CLIP mean/std) encodes the image, the BERT tokenizer encodes the question, the question encoder fuses the two, and the decoder generates the answer text. Nothing is trained or adapted here. What this repository adds is packaging: `verify_snapshot` and `stage_missing_files` (manifest digest checking and fresh-clone staging), `BlipVQAPipeline.from_pretrained` (verified local loading with `trust_remote_code=False`), `answer` (input validation, whitespace-collapsed question, greedy decoding under a caller-owned `max_new_tokens`, a `truncated` flag), `normalize_answer`, `exact_match`, `anls`, `vqa_accuracy`, and the `validate_inputs` and `evaluation_report` stage helpers.

#### Intended Use and Limitations

The uses below are the ones the package was built to support; everything else is either out of scope (§Out-of-scope use cases) or prohibited (§Use cases).

###### Primary Intended Uses

The task is open-ended visual question answering with short generated answers: input one image (`PIL.Image.Image`, any mode, converted to RGB), one natural-language question (up to 256 characters) and a token budget; output one short answer string, the number of tokens generated and whether the budget was exhausted. Envisioned applications are interactive image lookup ("what color is the car?", "is there a dog?", "how many people?"), attribute and presence checks over photo collections, accessibility aids that let a user ask about a picture, and prototyping of multimodal assistants — with the answer checked against the image by a human or a downstream rule. Within DIMER the pipeline is an inference component and a zero-configuration baseline for VQA, not a certified answerer for any specific image domain.

###### Primary Intended Users

Intended users are machine-learning engineers, computer-vision developers, and data analysts integrating visual question answering into research prototypes, internal tooling, or the DIMER workbench. A user is expected to understand that the output is *generated text* — it carries no probability, no correctness signal and no location in the image, and the model answers every question, including unanswerable ones, with equal fluency — that answers follow VQA v2's style (one to three words: a colour, a count, a noun, `yes`/`no`) rather than sentences or reasoning, that the token budget is theirs to set, that the model was fine-tuned on everyday photographs with English questions so drawings, documents, diagrams, non-English questions and expert domains are distribution shifts, that greedy decoding is reproducible on a fixed device but GPU and CPU outputs need not match, and that accuracy can only be measured on labelled question/answer sets they supply. Users who need text reading, answer grounding, captioning or batch throughput are expected to know none of that is provided here.

###### Out-of-scope use cases

1. **Capability boundary:** no answer localisation or evidence region, no confidence, no abstention (the model cannot say "I cannot tell"), no text reading (BLIP-VQA is not an OCR model), no captioning (a separate checkpoint), no batching (each question is a full encoder pass), no sampling or beam search, and no counting, spatial or commonsense reasoning beyond what VQA v2's answer distribution rewards — the smoke run counted `2` trees where one was drawn.
2. **Input boundary:** `answer` rejects non-PIL images (`TypeError`), sides below `MIN_IMAGE_SIDE = 16` px or above `MAX_IMAGE_SIDE = 4096` px, empty or non-string questions, questions longer than `MAX_QUESTION_CHARS = 256`, and budgets outside `[1, MAX_NEW_TOKENS = 32]` (`ValueError`/`TypeError`). Every image is resized to 384×384 without preserving aspect ratio, so thin panoramas are squashed and detail below roughly 1/24 of the shorter side is lost; fine text, small objects and subtle colours are outside the model's resolution.
3. **Input boundary:** the fine-tuning data is VQA v2 — COCO photographs of everyday scenes with crowd-sourced English questions and ten short human answers each. Drawings, screenshots, documents, charts, medical or satellite imagery, non-English questions and questions whose answer is a sentence fall outside what the upstream authors evaluated and what this repository measured; results on them are undefined, not merely degraded. An image with no content still produces an answer (see §Risks and harms).
4. **Decision boundary:** not for autonomous decisions that act on answers — content moderation, safety or compliance checks, medical or accessibility-critical descriptions, surveillance or identification — without a human comparing the answer with the image, and a locally measured VQA accuracy on the deployment's own labelled question/answer sets.

#### Factors

###### Groups

This pipeline is human-centric wherever the image contains people and the question asks about them: VQA v2 questions include "what is the man doing?", "how old is the woman?", "what race is the person?", and the model answers them from COCO-style photographs whose demographic composition (skewed towards Western, adult, lighter-skinned subjects per published audits of COCO) and whose annotators' conventions shape the answer distribution. Neither the upstream authors nor this repository audited answer accuracy or answer *content* by the depicted person's gender, age, skin tone, disability, dress or setting, nor the pretraining web corpora for stereotyped image–text pairs; the model will answer a question about a person's gender, age, occupation or emotion with a confident guess and no signal that it is a guess. Non-human groups whose accuracy is unknown, not known to be equal: non-photographic images (the tutorial's cartoon is one; the model miscoloured its door), non-Western scenes, objects and clothing, low-light and low-resolution captures, and questions phrased in non-native or non-English forms. An operator whose images contain people is responsible for a fairness audit on their own image set, stratified by depicted group and question type, before relying on the output — and for deciding which questions about people are permitted at all (see §Use cases).

###### Instrumentation

The upstream pretraining "instrument" is web image–text pairs at whatever resolution and quality the source sites served, filtered by BLIP's own captioner/filter, and the fine-tuning instrument is COCO photography (consumer cameras, Flickr uploads) with crowd-sourced questions. Inference images arrive from whatever produced them — a phone camera, a screenshot, a render, a scanner — and resolution, exposure, colour balance, compression and aspect ratio all change the visual evidence; the fixed 384×384 resize discards aspect ratio and detail regardless of the source, so a 4096×4096 input carries no more information than a 384×384 one. The pipeline validates type, size and question length only; it cannot detect an unusual capture, a non-photographic image, or a question that the image cannot answer. The synthetic tutorial scene (flat Pillow shapes, no texture, no lighting) is itself a rendering instrument unlike any COCO photograph, which is why the smoke run's two misses are recorded rather than treated as anomalies.

###### Environment

Operating environment: Python 3.12 with `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`, float32 on CPU; CUDA is used automatically when visible (float32) but was not exercised for this card. Measured on the reference machine with the GPU hidden (`CUDA_VISIBLE_DEVICES=-1`) and the Hub offline (`HF_HUB_OFFLINE=1`): `verify_snapshot` on the 8-file, 1.54 GB snapshot 0.82 s; load 5.6–5.8 s; one 640×480 drawn scene, ten questions at the default budget, 0.19–0.54 s per question (first call slower; every answer 2 tokens including end-of-sequence); a 4096×4096 blank image 0.66 s — cost is one 576-patch encoder pass plus a few decoder steps per question, roughly independent of the input resolution after the resize. Data environment: the model assumes the image is an everyday photograph and the question a VQA v2-style English question with a short answer; the synthetic tutorial scene violates the first assumption on purpose (flat cartoon), which is where the recorded misses come from. Documents, diagrams, non-English questions and unanswerable questions violate it to degrees this repository did not measure, and the pipeline reports no signal when they do.

#### Metrics

###### Performance Measures

The pipeline reports no accuracy measure. The answer is generated text with no score, probability or correctness signal; `new_tokens` and `truncated` describe the generation, not its quality. The repository ships three helpers because they are what a caller would use to evaluate: `exact_match(prediction, golds)` after normalisation (lower-cased, punctuation removed, whitespace collapsed) against any accepted answer; `anls(prediction, golds)` — normalised Levenshtein similarity `1 − lev / max(len)` over the same normalisation, maximised over the accepted answers, scored 0 below `ANLS_THRESHOLD = 0.5` — reported alongside so near-misses (`a tree` vs `tree`) are visible; and `vqa_accuracy(prediction, human_answers)`, the VQA v2 benchmark measure `min(number of matching human answers / 3, 1)`, which needs the benchmark's several human answers per question and is therefore **not** used by the report on authored data — with one authored answer it can never exceed 1/3. All three need labelled question/answer sets that the caller must supply; VQA v2 is not bundled. The public `evaluation_report(results, golds=None)` stage returns the report in machine-readable form: the `exact_match` rate and the mean `anls` over the questions plus one per-question entry (prediction, accepted answers, both scores) with the verdict `sample-sanity`, or the verdict `not-measurable` naming the labelled set that would be required when no accepted answers are supplied. The upstream paper's VQA v2 test-dev accuracy for its ViT-B models (about 78, upstream-reported; which pretraining variant the hosted checkpoint corresponds to is not stated on the upstream card) is not reproduced or claimed by this pipeline.

###### Decision thresholds

No score threshold exists in the model: it generates tokens until end-of-sequence or the budget and nothing is filtered or abstained. The decision parameter is the **token budget** `max_new_tokens`, default `DEFAULT_MAX_NEW_TOKENS = 10` (a phrase-sized budget chosen by this repository; every smoke answer needed 2 tokens including end-of-sequence, and the upstream README example uses the library default) with ceiling `MAX_NEW_TOKENS = 32`. A budget that is too small is reported, not hidden: `truncated` is true whenever `new_tokens` reaches it (a budget of 1 flags the one-token answer `dog` as truncated even though it was complete — the flag is conservative), and the caller should raise the budget and rerun. The evaluation helper carries one threshold of its own, `ANLS_THRESHOLD = 0.5`, a published convention from document VQA, not a value tuned here; `vqa_accuracy`'s divisor of 3 is the VQA v2 convention. Decoding is greedy (`do_sample=False`) with no temperature, beams or repetition penalty. A deployment owns choosing the budget and deciding how an answer is verified against the image before it is used.

###### Approaches to uncertainty and variability

This repository reports no central metric value and therefore no dispersion: the smoke run records timings, token counts and the answers on one drawn scene, not accuracy. Run-to-run variability comes only from floating-point kernel selection across CPU builds and accelerators; there is no sampling and no seed to set, so a fixed input on fixed hardware is repeatable, but because decoding is autoregressive a single differing token changes the rest of the answer, and CPU and CUDA outputs need not match; the drawn scene uses no text rendering, so its bytes do not depend on the Pillow build. On the drawn scene the model answered five of the seven authored questions exactly (`exact_match` 0.714) and missed two — `2` for one drawn tree, `red` for a brown door — one observation on one flat cartoon, not an estimate; on a blank white image it answered `dog` to "what is in the picture?" and `white` to "what color is the house?", and on uniform noise `tv`, which is what an uncalibrated generator with no abstention looks like. A caller who needs an accuracy estimate must supply labelled question/answer sets (several human answers per question for VQA accuracy) and compute over many images or bootstrap resamples themselves; a caller who needs a confidence per answer has none from this model.

#### Ethical considerations and biases

No external ethics board, red-team, or population-specific clearance reviewed this repository or, to our knowledge, the upstream checkpoint; nothing below should be read as implying one.

###### Data

The upstream paper describes pretraining on 129M image–text pairs from COCO, Visual Genome, Conceptual Captions 3M and 12M, SBU Captions and a 115M-image LAION subset — web-scraped pairs whose images and captions can include real, identifiable people, copyrighted photographs and stereotyped or offensive text, filtered only by BLIP's own learned filter — and fine-tuning on VQA v2 and Visual Genome QA, whose images are COCO/Flickr photographs of people and everyday scenes with crowd-sourced questions and answers; personal data in the training corpora is therefore present by construction. Neither was audited here. This repository distributes code, tests, and documentation; it does not distribute the 1,538,800,584-byte `model.safetensors`, which is staged locally under `weights/blip-vqa-base/` and git-ignored, and it ships no sample photographs — the tutorial scene is drawn in code. The operator must audit the images they submit for personal, proprietary, or otherwise restricted content; the pipeline performs no such check and will answer "what is the woman's age?" as readily as "what color is the house?".

###### Human Life

This pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, or housing, and it has not been validated or certified for any of them by this repository, the upstream authors, or any regulator. Foreseeable but unintended sensitive uses — describing images to blind or low-vision users where a wrong answer misleads, screening user-uploaded images for prohibited content, answering questions about people's identity, ethnicity, health or emotional state, reading medical or safety imagery — would be admissible only with human review of every answer against the image (the model invents an answer when none exists and gives no signal), a locally measured accuracy on the deployment's own labelled image/question sets stratified by depicted group, a documented budget policy with `truncated` handling, an explicit list of question types that are refused before they reach the model, and whatever regulatory clearance the domain requires.

###### Mitigations

- **Supply-chain integrity:** `MODEL_REVISION` is a 40-hex commit; `stage_missing_files` refuses a manifest whose `modelId`/`revision` differ from the package constants and fetches only manifest-listed files at that revision when `allow_download=True`; `verify_snapshot` then checks all 8 listed files' byte sizes and SHA-256 before any load; `from_pretrained` loads only from the verified directory with `local_files_only=True`, always passes `trust_remote_code=False`, and the smoke run loaded and answered with `HF_HUB_OFFLINE=1`. The upstream `pytorch_model.bin` (pickle) and `tf_model.h5` (TensorFlow port) are neither listed nor loaded. A test flips one hex digit of a manifest digest and asserts the loader refuses; another asserts a foreign manifest is refused; the import-boundary tests assert that a missing or tampered snapshot is refused before `torch` or `transformers` is imported.
- **Input integrity:** the public `validate_inputs(image, questions, *, max_new_tokens)` stage applies exactly the checks `answer` applies (both route through one shared private checker) and returns an input manifest recording the schema, the ceilings, the observed input, the checked questions, the budget and the verdict; `validate_image` rejects non-PIL inputs and sides outside 16–4096 px; empty, non-string or over-long questions and out-of-range or boolean budgets are rejected; `answer` raises on a malformed runner result; `evaluation_report` rejects mismatched or empty accepted-answer lists; `vqa_accuracy` rejects an empty answer set.
- **Reproducibility:** exact `==` pins in `pyproject.toml`; greedy decoding with no sampling; every result carries `model_id`, `model_revision`, the checked question, the budget, `new_tokens`, `truncated`, the device and the dtype.
- **Refusals:** no batching, no download without the explicit flag, no Hub access at inference time, no sampling, no pickle deserialisation, no attempt to guess whether the image can answer the question, and no filtering of question content — that refusal list is the operator's to implement in front of the pipeline.
- No statistical mitigation (class balancing, subsampling) applies: no training happens in this repository.

###### Risks and harms

- **Invented answers:** the model has no abstention — a blank image yielded `dog` and noise yielded `tv` in the smoke run — so an unanswerable question, a wrong image or a leading question produces a fluent, plausible answer with no signal; downstream consumers that trust the string (moderation rules, databases, LLM pipelines, screen readers) inherit the error silently.
- **Confident guesses about people:** questions about a person's gender, age, race, health, emotion or intent are answered from appearance with the same fluency as a colour question; such answers are stereotype-driven by construction and can cause direct harm when surfaced or acted on.
- **Plausible near-misses:** `red` for a brown door and `2` for one tree are short, confident, wrong answers; only labelled evaluation on the deployment's images shows the rate, and the tutorial's 5/7 says nothing about it.
- **Automation bias:** crisp one-word answers invite trust that generated text has not earned.
- **Truncation:** an answer longer than the budget is cut (reported via `truncated`); a caller who ignores the flag ships a partial value.
- **Bias amplification:** any image population or question type VQA v2 under-represents (non-Western scenes, non-photographic images, minority groups, non-native phrasing) is reproduced as uneven accuracy, undetected because no per-group evaluation exists.
- **Resource use:** a 1.54 GB model and ~0.2 s per question on the reference CPU with an encoder pass per question; a question-heavy workload scales linearly, and the CUDA path was not measured.

###### Use cases

Prohibited even where the model would work: asking questions about images of people in order to infer or record protected characteristics (race, ethnicity, religion, health, disability, sexual orientation), to identify, track, profile or surveil individuals, or to make or support decisions in employment, housing, credit, insurance, education, healthcare access, law enforcement or immigration; processing images the operator has no right to process, including intimate imagery and licence-restricted material; deceptive uses that present generated answers as verified facts about an image or as evidence; and any use that violates the upstream BSD-3-Clause licence terms (including use of the Salesforce name for endorsement), the DIMER deployment terms, or the consent and data-protection obligations attached to the images processed. Autonomous high-consequence actions triggered by unreviewed answers are prohibited by the intended-use contract above.

## Immutable provenance

- Model: `Salesforce/blip-vqa-base`
- Revision: `787b3d35d57e49572baabd22884b3d5a05acf072`
- Snapshot manifest: `weights/blip-vqa-base/dimer-base-manifest.json`, 8 files, `totalBytes` 1539754668
- `model.safetensors` SHA-256: `33786eed34def0c95fa948128cb4386be9b9219aa2c2e25f1c9c744692121bb7` (1,538,800,584 bytes, float32)
- `config.json` SHA-256: `689a09e2a9980b7fcad329271c032254fde23b1ee7a90c67b003e1867dc9c098` (4,559 bytes; `BlipForQuestionAnswering`)
- `preprocessor_config.json` SHA-256: `0aa66e2e9ac3ea3b5cd4388c35072e22db4e1cc1f96c7872bed07749c712ade1` (445 bytes; `BlipImageProcessor`, 384×384, CLIP mean/std)
- Weight format: SafeTensors; loader `BlipForQuestionAnswering.from_pretrained(<dir>, local_files_only=True, trust_remote_code=False, dtype=float32)` with `BlipProcessor` from the same directory. The upstream `pytorch_model.bin` (1,538,966,629 bytes, pickle) and `tf_model.h5` (1,539,707,712 bytes, TensorFlow port) are not part of the manifest and are never loaded; the executed artifact is the SafeTensors file only.

## Input/output contract

- `BlipVQAPipeline.from_pretrained(device=None, weights_dir=None, allow_download=False)` — stages missing manifest files (only with `allow_download=True`), verifies digests, loads; `device` defaults to `cuda:0` when visible, else `cpu`; float32 on both.
- `answer(image, question, *, max_new_tokens=10) -> dict` with keys `answer` (stripped decoded text), `question` (whitespace-collapsed), `image_size`, `new_tokens` (answer tokens plus the end-of-sequence token when generated), `truncated`, `generation` (`max_new_tokens`, `do_sample` false, `decoding` greedy), `device`, `dtype`, `source`, `model_id`, `model_revision`.
- `normalize_answer(text) -> str`; `exact_match(prediction, golds) -> bool`; `anls(prediction, golds, *, threshold=0.5) -> float`; `vqa_accuracy(prediction, human_answers) -> float`.
- Ceilings and constants: `MIN_IMAGE_SIDE = 16`, `MAX_IMAGE_SIDE = 4096`, `MAX_QUESTION_CHARS = 256`, `MAX_NEW_TOKENS = 32`, `DEFAULT_MAX_NEW_TOKENS = 10`, `DECODING = "greedy"`, `IMAGE_SIZE = 384`, `ANLS_THRESHOLD = 0.5`, `VQA_ACCURACY_DIVISOR = 3`, `INPUT_SCHEMA`.
- `validate_inputs(image, questions, *, max_new_tokens, names) -> dict`; `evaluation_report(results, golds=None, *, sample_kind) -> dict` where `golds` holds one sequence of accepted answers per result; `verify_snapshot(path=None) -> dict`; `stage_missing_files(path=None, *, allow_download=False, downloader=None) -> list[str]`.

## Runtime

- Pins: `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`, `huggingface-hub==0.36.2`; Python 3.12.
- Precision: float32; preprocessing resizes the image to 384×384 (aspect ratio not preserved) and normalises with CLIP mean/std (`BlipImageProcessor`, snapshot defaults); the question is tokenised by the snapshot's BERT tokenizer; greedy decoding.
- Measured 2026-09-14 in the Windows venv (`torch 2.14.0+cu130`) with `CUDA_VISIBLE_DEVICES=-1` and `HF_HUB_OFFLINE=1`, device `cpu`: `verify_snapshot` 0.82 s (8 files, 1.54 GB); load 5.57–5.83 s; `answer` on a synthetic 640×480 cartoon scene (sky, grass, sun, red house with brown roof and door, one tree, white ball, drawn with Pillow) at `max_new_tokens=10` → "what color is the house?" `red` (0.54 s, first call); "how many houses are there?" `1`; "is it daytime or night?" `daytime`; "what is next to the house?" `tree`; "what color is the sky?" `blue`; "is there a tree in the picture?" `yes`; "what season is it?" `summer`; "what is the yellow object?" `sun`; "how many trees are there?" `2` (one drawn — miss); "what color is the door?" `red` (brown — miss); 0.19–0.21 s each after the first, every answer 2 tokens, no truncation; `evaluation_report` on the seven tutorial questions against the authored answers: `exact_match` 0.714 (5/7), verdict `sample-sanity`; 384×384 and 4096×4096 blank white images, "what is in the picture?" → `dog` (0.66 s at 4096²), "what color is the house?" → `white`; uniform noise → `tv`; `max_new_tokens=1` → `dog`, `truncated` true.
- Tutorial execution: `tutorials/blip_vqa_colab.ipynb` ran top-to-bottom in a fresh local kernel (all 8 code cells, 148.1 s including the 1.54 GB staging, same seven answers as the smoke run — 5/7 exact with the two recorded misses); recorded in `docs/release-verification.md` as pre-flight, not supported-runtime evidence.
- Tests: `pytest -q -o addopts= tests` — offline, no weights required; `ruff check src tests tools` clean.
- Not executed: CUDA path, photographs (only a drawn scene, blank images and noise), non-English or non-ASCII questions, answers longer than the default budget, any VQA-accuracy measurement against human answer sets, questions about people.

## References

- Li, Li, Xiong, Hoi. BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation. ICML 2022. https://arxiv.org/abs/2201.12086
- Goyal et al. Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering (VQA v2 and its accuracy metric). CVPR 2017. https://arxiv.org/abs/1612.00837
- Biten et al. Scene Text Visual Question Answering (the ANLS metric). ICCV 2019. https://arxiv.org/abs/1905.13648
- Upstream code: https://github.com/salesforce/BLIP
- Upstream card: https://huggingface.co/Salesforce/blip-vqa-base
- Transformers `BLIP` documentation: https://huggingface.co/docs/transformers/model_doc/blip
