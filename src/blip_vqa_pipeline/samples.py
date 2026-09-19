"""VQA dataset contract for fine-tuning: the pinned VizWiz-VQA sample, validation, seeded image-disjoint
splitting, BYOD loaders and JSONL export.

The default dataset is **real** and out of the base model's distribution: VizWiz-VQA (Gurari et al., CVPR
2018;
CC BY 4.0) — photographs taken by blind people with a spoken question each and ten crowd-sourced answers,
about
a quarter of them judged unanswerable (blur, framing, darkness). `Salesforce/blip-vqa-base` was fine-tuned on
VQA v2 and Visual Genome, never on VizWiz, and it has no notion of an unanswerable image. The annotations come
from one pinned parquet shard of the Hub mirror `lmms-lab-encoder/VizWiz-VQA` read **column-only** (the shard
holds 864 questions with their full-size images in one 406 MB row group; only the four text columns, about
0.13 MB, are fetched over HTTPS range requests through `pyarrow`, the shard's declared size and SHA-256
checked
against the pins before any byte is read and the decoded columns' SHA-256 after). The 240 questions of the
sample were drawn from those 864 with a seeded shuffle at build time; their images are fetched one by one from
the VizWiz project's own image host with a per-image size and SHA-256 pin, so an image that differs from the
pinned bytes is refused, never silently used.

A record is ``{id, image_id, question, answers, category, image}`` — the ten accepted answers, the VizWiz
category (`other`, `yes/no`, `number`, `unanswerable`) and the path of the digest-verified image. Every
record's
image is its own (VizWiz has one question per photograph), so a split by image is a split by question.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import re
import time
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from .pipeline import MAX_QUESTION_CHARS, MODEL_ID, _check_inputs

CORPUS_NAME = "VizWiz-VQA"
CORPUS_REPO = "lmms-lab-encoder/VizWiz-VQA"
CORPUS_REVISION = "d428a2dae984f79cf1b9d99467dfa883e0c30686"
CORPUS_RELEASE = (
    "VizWiz-VQA v1 (2018) as mirrored on the Hugging Face Hub, dataset revision d428a2da (2024-03-08)"
)
CORPUS_LICENSE = "CC BY 4.0 (Gurari et al. 2018; vizwiz.org/tasks-and-datasets/vqa)"
CORPUS_COLUMNS = ("question_id", "question", "answers", "category")
CORPUS_FILE: dict[str, Any] = {
    "path": "data/val-00000-of-00005-7775fd61bc6a3d98.parquet",
    "bytes": 405_819_917,
    "sha256": "62a5fddb577d9165c3b89d6fb565cddfb4e7d41294ca16373de6263baff7b36d",
    "rows": 864,
    "column_sha256": "00b476ddd96ff74ee65de12781df24c96f88374e36f8f9cc1fa75987012f8e3c",
}
IMAGE_HOST = "https://vizwiz.cs.colorado.edu/VizWiz_visualization_img/"
DEFAULT_CACHE_DIR = Path("weights") / "vizwiz"
SAMPLE_SEED = 42
SAMPLE_SIZE = 240
SAMPLE_SPLIT = {"train": 140, "validation": 40, "test": 60}
MIN_RECORDS = 8
MAX_RECORDS = 5_000
MIN_ANSWERS = 1
FETCH_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
FETCH_MAX_ATTEMPTS = 6
FETCH_SPACING_SECONDS = 0.25
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
# question_id -> (sha256, bytes) of the pinned image file on the VizWiz host, for the 240 sampled questions.
IMAGE_PINS: dict[str, tuple[str, int]] = {
    "VizWiz_val_00000005": ("745b3a68baf9d5ce039aa1efe805869b5127f61dacc46fed6db4d6d850f95d10", 193522),
    "VizWiz_val_00000008": ("47a17987d743e8e882477707541929f0f6f9b1e234b9325713768614c6d8bb33", 1597479),
    "VizWiz_val_00000010": ("279a7f43bc247e6bf2c49f0fe670262a009503f968caf07654aaab495b8ebc5a", 257286),
    "VizWiz_val_00000013": ("ffb959ac03562a779dd58fe33988952a08484116fd070b90b3f90809b3eb7de0", 243408),
    "VizWiz_val_00000016": ("fa443fca468aa6b1b14cf6fc9dedb894672f090520be3aea5bb96bd8bd3dae70", 423596),
    "VizWiz_val_00000017": ("43fb47df8655a67bbdd59e0d148cdb2b79c643c8a3f3eb681d7f6e6771462cd7", 411585),
    "VizWiz_val_00000024": ("97244f2257a09e7abc06872852639aad2fbd1bba588bb86e8419aee5f70c8df3", 450615),
    "VizWiz_val_00000028": ("b2932cc756e3e34150bf8fbaf801129b402183a4f6d2f7466dc44342484c6135", 1240097),
    "VizWiz_val_00000029": ("cd39333f526285e8a4d5c80146c817f90324358f957260c9a367bda15357c7ad", 35243),
    "VizWiz_val_00000031": ("69d1e14026c6e94cb32962a8f8c1f3b17e48c46fc77217686e35c47a87da98e9", 236006),
    "VizWiz_val_00000033": ("b92bb8363cb0ae94bb4efbcb155b50d4acdad22727339c1eade0addeaa2109ad", 144217),
    "VizWiz_val_00000034": ("f0dab30bc1b83d08700987d471b5d1d0144663365c89bcef4320c857f3d38a47", 405285),
    "VizWiz_val_00000036": ("be5632606bcc41dfb6f607b7b3c54da3b7a771a33be58e2f5b2a3219fd3ad3ac", 335298),
    "VizWiz_val_00000039": ("3ce7561cc3d0d9bb77b13d7d24e0d76d161469d2ba8ae828e6e51d023085f028", 1111181),
    "VizWiz_val_00000041": ("d8ef8e546ca1e0ca5dc92ef8496f638eb22c8a269b8344ef7743bcc28ce1b0f1", 365576),
    "VizWiz_val_00000042": ("013a6a9c310ffe4af49db1950be7cb7086c4fbee4113e0201de0757c3e93e752", 395265),
    "VizWiz_val_00000045": ("35f85b68b21bfea8efc68e9de4ea66192be2f395aca9dc342e207950a2dadf9d", 504358),
    "VizWiz_val_00000049": ("03a1e13011702db365b11a805c163249b1788902b332e323ca475ff6e5f1734b", 1267816),
    "VizWiz_val_00000050": ("2d6f58cc4ac712609f221a17bdacef4d6751c90a329e5ddce375e7c005bbd95a", 223091),
    "VizWiz_val_00000056": ("427c4f6d7014dc51dbfd142ba1eccbb29341abf29a2f4f3dbbce3452100c26b2", 90153),
    "VizWiz_val_00000064": ("46b3198d80bc4a3146ae8d6d8dc92859def2020732e6acc7a437af920e0027d2", 2501),
    "VizWiz_val_00000078": ("be7e5770c37bac4261ba83baa8ef5cd152407dafe113f2e390a742c9121a272d", 290526),
    "VizWiz_val_00000084": ("d6349eb9b1a68bb1754706af38392191b2a4d31e320c659371c9c7bafc97166b", 1021687),
    "VizWiz_val_00000088": ("ba6d0ec03ef5f6d52081703ac8aea68f8ded4161ef7e76da082e0f7b24143fda", 359623),
    "VizWiz_val_00000092": ("df20fcd240cf179457f5b8055c393d02f8a50808b011ecdb54b2b3f977175ceb", 435333),
    "VizWiz_val_00000105": ("70ad4b130fcc571d5fc94833612de3b7497b706964332c050043557259d715e2", 427776),
    "VizWiz_val_00000106": ("2ecac5a9af9997e7246b94353ce95c0937ebcd72f1692d6f8e28352ffde242ca", 271080),
    "VizWiz_val_00000110": ("c6450ac8eedce3b59f7ea803cdf0a2f05d1317ce6cefae14892dbf49cfd2f10e", 401312),
    "VizWiz_val_00000118": ("cde2330d2873fdb011c10f599621674ce8db9192ea2f15c7fbd6ee4b94c20c7e", 187545),
    "VizWiz_val_00000119": ("4120f1b7d9ec037dc8890fc65e1716cb14c8f7d1796091bde4fc66675880716d", 2051608),
    "VizWiz_val_00000129": ("b86e56ce16d518c4ee5f63b90dbd53290b0610528acc6c530c0d87c36d0b8107", 240203),
    "VizWiz_val_00000137": ("f49dc77f1645d7fcc12666bfbeecee68a561cd5d8f574c7fcacb1838503b3544", 546814),
    "VizWiz_val_00000138": ("c9fa0fb650d506387cb36bbdf1ea34b4a2faee5c9f9804acdaa3e5d8d4424484", 662194),
    "VizWiz_val_00000140": ("80fa56f2a4c88837d1191fa2c91d6334066e6d769c931b39a47d2523765eeee6", 2572125),
    "VizWiz_val_00000147": ("2556f76eb17f600e57c3ffe1b9f9160eae9dd9e9b592358a89badf16cad0db2f", 28332),
    "VizWiz_val_00000148": ("f02199f7215d1540ddda879e907442f5f65ad0f30ccd03d3b8bb47a134f17ed0", 225420),
    "VizWiz_val_00000149": ("193a2ea059f87a4ee3f1457753ec8bd510b49583ba7f28db72e9e136c6ad95b1", 202461),
    "VizWiz_val_00000150": ("c03d5d0e03e265c3b38df6b05e2f0893f683109890cff63df6b8d5b7029bcb44", 262091),
    "VizWiz_val_00000152": ("a980240b11644a6a5f6518879df043e259cb567fc0ff321523ae303ab8adcb36", 232218),
    "VizWiz_val_00000172": ("3ce308b218e55aa53e61fa1dcfc05cf8aad5e58f30d0be82dc667de63a774461", 255860),
    "VizWiz_val_00000173": ("bd59f38da5e1a7017dcd0d041581cc0fc97da675e9ce506aed9ca04e703fb32a", 306305),
    "VizWiz_val_00000174": ("d54a95cd198260e4c1188ed2d9465dab16bf76cb810e55a1cbd7698e0d333a47", 2100400),
    "VizWiz_val_00000177": ("31cafbc5b44780daae17fc1becaa9dc0959f20fe123fa00aa2b086beac608628", 174310),
    "VizWiz_val_00000178": ("5d7a52878738032a280220b0aa8ddd328d9e585dc0b7b28ae8c0846716a7935d", 550231),
    "VizWiz_val_00000180": ("0bebbaea86c434a17c4ccb6ea7b47e74a74c9750635b7d30364752a7559851b3", 2031964),
    "VizWiz_val_00000184": ("b7cbe07880d20fce18cbeeafcbf16c7e81a2b02f7ce9e723413d0e2aa0cd59a6", 293399),
    "VizWiz_val_00000185": ("81395482acace4a6f3d5413d7a73702d890f30da3d13be0ca47871da81488f37", 2219),
    "VizWiz_val_00000188": ("cd822df62e292704a01fad92984a0e19fac7726ac92536e6a325791032e20402", 117216),
    "VizWiz_val_00000191": ("2add1b667f5bd492e57b59e8cc31686e7693c075ca9215d5d353d9eaddf08986", 613376),
    "VizWiz_val_00000192": ("d4a02356af45cb348eed78c78b9f106ab300f8994b0d842aef8d9d76e0c47247", 241807),
    "VizWiz_val_00000193": ("fb9ab63f4af0d7e3064a80ec33c8a3fc8c9aa4c520aaba5aa7e0b074f760403a", 2251907),
    "VizWiz_val_00000198": ("56c325cf4b751245c81bed670655174292f4b12a1d1af09ddd282031e7ae2b2f", 471059),
    "VizWiz_val_00000200": ("9a8bd1e3f9165ed4b636982b97a6316d3c4f7a22ab4054fa5c177b25c055d336", 1912818),
    "VizWiz_val_00000201": ("3b143d15ed81f8194d56408ee31cf61cb7ed3054f4e8a23e6a4e91fc34451dde", 371093),
    "VizWiz_val_00000202": ("04fba3896a01731a4ade766e20b7fedadffd127e0de12f1bb3aee82d511aa1a6", 73397),
    "VizWiz_val_00000213": ("3c7f10866d95ae85a776c9ae7ee80c94424e96b560f918adb64daa80245e005b", 262222),
    "VizWiz_val_00000221": ("d1df994b8b7d75e737613b97459f63bdf2961b8200b1f5f12bf2b51694f6fdc9", 1708749),
    "VizWiz_val_00000227": ("69037a7bf3b15bc170a01235f007056d29176fe5338ba4c9277f84b7585a24d8", 255590),
    "VizWiz_val_00000229": ("dd1dbe686f38e10aee881234f211bd35bb2205c01c87edba09a14c414802761c", 194953),
    "VizWiz_val_00000231": ("6e65a8342b7ea04ca26409053ad22597b24820eea1ba7d5af611716c02336916", 1551609),
    "VizWiz_val_00000232": ("6484c1f9ee28af578affea2828bc2a07b07e14aa4f321b33b6c5aefb97e488b8", 385300),
    "VizWiz_val_00000236": ("9adce1f6dd115687ccef958b8abab2209dff8238c6e138372d60f459a3e99a50", 412800),
    "VizWiz_val_00000239": ("2ef00a154d4d8a0167643062d61c719a57874b090c2a3b03669e51bf5ba9a541", 210342),
    "VizWiz_val_00000241": ("509288565451e30e4ef310fff75f73dbc7be0661b18c6f0862716adac88db0e3", 1952709),
    "VizWiz_val_00000247": ("b88edbbf34e65ede2a351bf77bb223162b6da7596ad87977b7d67eb2fdcb143e", 326439),
    "VizWiz_val_00000249": ("35664da9502b472ac14f19e074ec8226b91fd09c0c8010f73a3164da87676824", 350493),
    "VizWiz_val_00000251": ("2a7a595c2b353e6d22fc6f93ca9c2c25caa9364f485ad70592f4fe4aa6352b1a", 259315),
    "VizWiz_val_00000256": ("c54e83baa269ed8f89b9b8a0aaf1b5e308718f8b78f38837d4c683263f6c86bd", 474915),
    "VizWiz_val_00000262": ("0f0f7d29309d5d24b318a996e6a99668f1daa4d6097b35fb6c42b908f6021f75", 519161),
    "VizWiz_val_00000264": ("1c49bda37156cdf8ace1e84c8d0ef25466bf2531c23e6005aee58517d96fa1ac", 1162217),
    "VizWiz_val_00000268": ("769091fc8056ff053bab6c1559eed97ad717f695b0fe72277286a99d8b76f1a5", 229150),
    "VizWiz_val_00000277": ("73d72c9b44e62e0a21ac0afe8b48e1fdaf700bed72b98ea16520422d92cb944f", 271242),
    "VizWiz_val_00000289": ("5e772d6d7f25a3f8a4c5bc0f279919af57e1c9f0da6a29dcff936426f6cd6365", 384901),
    "VizWiz_val_00000290": ("869250e5b633bfeb02d035135b24aad6973c1b4b5e36818a975ee452e5fbde44", 468287),
    "VizWiz_val_00000293": ("a8af72b3723b00276472c716b9a1e2ed9f52add12155adbd4ca32970c95a8518", 380239),
    "VizWiz_val_00000294": ("b75b0eca91add80df93a9ae6c7b5e786daf607cf73090e128ec6d2fefe52bbd0", 271019),
    "VizWiz_val_00000299": ("84e58873765084ba785c0ddd8b839cc8ee3aaf3e3c1001a81e965efacd0d3dd1", 261444),
    "VizWiz_val_00000301": ("0132a60fa7e224742903e4c9132007812b663fde18fe7091e785e78580e2a143", 341539),
    "VizWiz_val_00000302": ("0c7e659a2bed020dda5e3601905141b701afc8939eca06df77e58cf06377973d", 2424785),
    "VizWiz_val_00000304": ("1538bc77e391a08e2fb239aefd15fe181565836d5dda878310140f423182c20a", 180445),
    "VizWiz_val_00000308": ("a03f98730e6265ac66b9f25a201bf2acfcc5d582cdfefb8b4c190fc249d96eb7", 282583),
    "VizWiz_val_00000311": ("f4feef3ab8647a06de1b2fb4d77e701379cdbf98773dcdc24c1f273dc97b7f32", 283623),
    "VizWiz_val_00000312": ("7a012c33494f40ce6c54ba6d679e293ae7fd5a2308659207a64c093b32790200", 391424),
    "VizWiz_val_00000316": ("e5aafaf20cd0a5332fe2d151f6d6e6c11f9e164b430fc0e6a309caf4ff5389f3", 1021927),
    "VizWiz_val_00000318": ("04551b32870d1be34b2c4ebeaa231dbc1f20340714be8d6e71c98bd00f58dd46", 486112),
    "VizWiz_val_00000320": ("3ae7cf77417420770a0689aded2713bdd3dd588d627de151d8522d0f4b480638", 1391949),
    "VizWiz_val_00000325": ("9baa1bb11b6908a86b2ed0f25dac232bbedeb6a09ac6457cedf5cdef31f88d50", 332642),
    "VizWiz_val_00000329": ("6ce3bd3e4fb8c26d6fe6313221e74dfd867e856c520eff5b6ce1150499cc3f6b", 425180),
    "VizWiz_val_00000330": ("b9f1c7ed1dd25a63a54282298839ef67d4235ff96b26c331a690cb9189c348a6", 19147),
    "VizWiz_val_00000333": ("a49d96b711ecd949ba7fa5811bae5d0c424bd4a190d63b6c2d672578c75fc85e", 1609251),
    "VizWiz_val_00000335": ("a5517f2bc64aab4d7ffd8cdefdbf52f00c5a08505eab14add65a0f14ddc167d9", 439710),
    "VizWiz_val_00000350": ("cdbc6553a6ffac027aa1c439571a4ad744007d6e8282afe9d197dd4e18531e0e", 426073),
    "VizWiz_val_00000358": ("749186e3a3ac8887aad0e83fc06d36bf6c4621335797b812435df1b3731f76ea", 52153),
    "VizWiz_val_00000364": ("936aea10b38a50b5dde2d71a6b88da49714ee5eb7dc32a94ece84d3154f25e81", 389493),
    "VizWiz_val_00000369": ("c0ca0b9217908ffaae8b7349ab6be7cf0ae8702822073af4bd67791139c582d9", 323265),
    "VizWiz_val_00000374": ("2cd343e4a2a42c7daf3a0b9347930917d515be744b9640bbd40f04753a347660", 388355),
    "VizWiz_val_00000375": ("147b1ad3b8d65a28a5aa74221d1cc74aa4042522723d1e92a3b178c3ac551d21", 553543),
    "VizWiz_val_00000378": ("b2edd74ec94fad2de20181a68080bdf8091fa03a2501dcc372277358ceed582b", 268619),
    "VizWiz_val_00000381": ("f3970ef5904ddfd1e292f5b525f84f4fe4b0df57c18eb80720202961405ef567", 466223),
    "VizWiz_val_00000384": ("83f7207d7a629150ce67c2ffc01454a2ede23c06ac4220dda17c62f13a502b33", 373040),
    "VizWiz_val_00000385": ("32c3ff249e3f02335a2e72aa0612f64bb71853c8a5a17ca9ae52c57b215481ba", 346097),
    "VizWiz_val_00000392": ("659ff80507a9aaa0531204bc3aa4e3b64d89987a12e49c6b5568e288143a08ad", 46558),
    "VizWiz_val_00000396": ("06d553662eec3dfae3c320981f7c43cc3a42650349149e55e983d0167eddadad", 515956),
    "VizWiz_val_00000401": ("4b63b767910e54899a3539f1a9862ae30de1bf29b09ffe7135c20b62b3dc03cd", 412706),
    "VizWiz_val_00000402": ("b21d2fbb6dc789dc1329124c02fe545a4cf6eb70cbc439970a20ebac7662e4da", 592976),
    "VizWiz_val_00000406": ("140275039e0511be2d1dcc5209929bd07c3fc3474e53aa09f0f20f2e2be89432", 529333),
    "VizWiz_val_00000409": ("da37988fef187b87b2bb0f706cdc4c4bec6bb61c9b68a3423756d433b4c0aff0", 308345),
    "VizWiz_val_00000411": ("9879ce0eff736b80c6c649a88caa356a9b74c9dbf7520156c596d20432df5f38", 158435),
    "VizWiz_val_00000415": ("89e0bf14c0f115572c2e7dce1f378618d7d0809760d35516f9540cb69910f20d", 441884),
    "VizWiz_val_00000417": ("2914a091bfcbc5e6856d2e49361475db61c05002e825331008281f861f691446", 175405),
    "VizWiz_val_00000422": ("5be76cda6214a66aa396ba558b10e9eb0582fadf19662cecedffb16374a44f06", 917382),
    "VizWiz_val_00000423": ("d839f2b3289149e78bde2ea3f07c5cc4ec0c5b44ef7ac4a7f27b6b654edeb775", 528143),
    "VizWiz_val_00000431": ("5e4b7c8141cba22c50df88dccfcf1074b377cdbb483cda13e91eb2d1a0f9b343", 247846),
    "VizWiz_val_00000434": ("e9eafae8ad2e186198cf67b2138410ab73a5d6410c4f3e78007b231306f0de76", 506106),
    "VizWiz_val_00000435": ("d5cb2f2e608f042ee4f3507df1ebe4ee553355cfea652895b9c7029a7532c988", 331851),
    "VizWiz_val_00000437": ("40bea01f913deddefc17c3768bcc6606db678d167875fa61eedf4f08151823e7", 383075),
    "VizWiz_val_00000439": ("8147af81ba20a7ec2bf8163f76c441c28ff1ae300d906774d4952ab123c3b071", 307212),
    "VizWiz_val_00000440": ("acfc3fb1b02c8f7587444a4d6ff6491a930cadf56534f7b0e476bfd585f149c0", 141494),
    "VizWiz_val_00000442": ("46f18a75d997c2a8f049d3d1018275531222ccdc9ff98f5e2da3dac763268bab", 56531),
    "VizWiz_val_00000452": ("195f87c9319923f2e410d80ba9dd0540febca990f9c25574b01a1e6bcd044ac6", 41996),
    "VizWiz_val_00000454": ("f5582bb721cbc7c75637976656409cd772642d99114e4f78708711cbfac826bd", 340333),
    "VizWiz_val_00000458": ("c678ab29652cb8650c5ecb7a90d6ca2df8308517b45b9bba4f8c4610f7882294", 468227),
    "VizWiz_val_00000461": ("9e48cac492a9d807d55fbc0fdd3d2ddeb7d013f262bb0fa5bb7a27d8ca0c47d8", 545792),
    "VizWiz_val_00000463": ("fd32de23e473af7e76d37a0fd30c8a818dea1e91a473a433c6bbb19fe443fece", 193048),
    "VizWiz_val_00000467": ("86bbf5549a356758cd816512e6fe059d869cc9abf2c9fbd2abcb9228b00c0325", 361799),
    "VizWiz_val_00000471": ("ef645547551f71395625dbdc4991d522aecdb29d13ce867ba77bfffc3299c303", 209653),
    "VizWiz_val_00000472": ("f3474d2e43fe0a327fdec1a47fc53ef9116f8aadb68ec7ffbda8fc4b8e05e1b0", 231100),
    "VizWiz_val_00000474": ("1dd8686210182ce8974bffa57e037febb4a4af4e2743d53f506b13cd8fac9383", 658476),
    "VizWiz_val_00000475": ("83ae67ced8f8b06255dd0a0a41c7834f372a6f3dfdd4bc82674854b2d0b9e462", 448609),
    "VizWiz_val_00000476": ("06317f05a019aafae5c6c790cbcb35b4a4eb115a01b30a6937a4fe29c4564ce7", 412706),
    "VizWiz_val_00000485": ("35070fbb1c2c89e704ae87bc651d464f3e1cb28b9b44bfdb86afa4b68c46443d", 39949),
    "VizWiz_val_00000487": ("db3dcab25c00739fd52f77cc14bdc4835003946351de17a301e00bbb825a2963", 406717),
    "VizWiz_val_00000495": ("233fc48b1d0735e3afc23ed4802e5c680ba5e49c3fdde7eedcf067e6cab6320c", 513190),
    "VizWiz_val_00000496": ("a16b2ab40258c77de4c6c1c7a5b358e386be903248a42b5af9209bc89b49c5e0", 222319),
    "VizWiz_val_00000501": ("a1b799bf7e95139ec3aa5f1ba68ab856b2672cfcf3fed6c24ad033dad64c8ce3", 390077),
    "VizWiz_val_00000502": ("7d6b20a0f1800f01fd2dc4e34c00b9525e699a93f1d0b6615aade6f1e5927c97", 358756),
    "VizWiz_val_00000503": ("b6fa611f68b05201de68e525e05d6d40520886aa447e2556656900c93bb13f21", 462972),
    "VizWiz_val_00000504": ("78c60c594e936592b6feb0c84b9dcde5ca21f85c99a5de6b29f446a5e7fd3f93", 573398),
    "VizWiz_val_00000508": ("dae0e8b4299d676dab7c16a445e242e70053563c5b6e85c7f76d272a07f2472a", 163991),
    "VizWiz_val_00000510": ("f87e11f84e9a066cd2143c05fcb82a830d9dc907dd7b9c149bcfda85f523cfdc", 576431),
    "VizWiz_val_00000513": ("2d4dbedb845d15fdd3ede54e54ece4de48055ae5ae16e80d1a86c273db158083", 46640),
    "VizWiz_val_00000518": ("e114c0ab8f1b2ac2f0a0126966084f456342816665dd235f9cf0a32d713cb1ea", 291555),
    "VizWiz_val_00000527": ("b52dfcdd10ab1c7066df037c6160f45fed1c75f7eac325b7ad7e515de4619651", 447622),
    "VizWiz_val_00000533": ("f101dbf8cd1c3b4c4b1de8b500ffd35398065a16a21078b511fe5872428f5abb", 687424),
    "VizWiz_val_00000534": ("d3a0b6650ca3df499cd9976db04d6ed1551d7177d0cae44b67820ae2e7771244", 655698),
    "VizWiz_val_00000535": ("cc0b761a5cd1fe5e2dddf909093d5e25bc072eb4ffec9b699ca79ac4ae0056f5", 493116),
    "VizWiz_val_00000542": ("ea2c3fc1504a1da8ccaea65f3a5b33e6cb087880c55ae6828f0a67c56a2e71f7", 429894),
    "VizWiz_val_00000544": ("e34bdc02a0ba207019f53b8e66d93c2f95c66e181f995bf8dc285d9b296f2722", 327210),
    "VizWiz_val_00000547": ("1b4cdfb3cb8b9d59ebbd1c7065d8ca3a545681eb6338614197cd81acfda7a50b", 628198),
    "VizWiz_val_00000548": ("8150e51b5e617c639474406bc378131ec1e4e6c5d0c8eca3763a04e33ac4efba", 575236),
    "VizWiz_val_00000554": ("6ee6667d206fae8834f881710e97813d2f22eb164e4f33c481da76c41322a3d2", 769406),
    "VizWiz_val_00000557": ("f7176bd59aef2858c23284f80e9cf1e524d6f922be34908062a2ec7bfdb81f1c", 387730),
    "VizWiz_val_00000564": ("7aa8a64d11b9aa18c5cb14895372fc156ad2b2be7081e20941cf3d08e6665d75", 464643),
    "VizWiz_val_00000569": ("5169167c5561b49fedc25b36d8cbe1c06ec40a068928e314f9e175b6f5ce86bd", 55216),
    "VizWiz_val_00000571": ("d4782f1986446b396e6a0a4545b873a029bbcfed6fa890d29233c495b2859126", 278039),
    "VizWiz_val_00000575": ("4797579893a656283daf09a85752668c2c67ca4857754f73982b681c289dbb25", 447663),
    "VizWiz_val_00000581": ("0dc9dcab5580d6ce80fe36f455841dd64b4742172e9f555b490633f589042269", 141342),
    "VizWiz_val_00000582": ("f38c3eb68dccf123cd3a38ec6e8ff3a536c0c3b5f7a7482ca056aae68c1f98d7", 281311),
    "VizWiz_val_00000585": ("d706256eb31c4ddb8d4434d27ce97e1ba2b65d103c08f82dbafccc30f5fde898", 517522),
    "VizWiz_val_00000586": ("83fb17f0b90a0a554d714cd1a48bebd322cc0f0260555e50cb06bb01a42b948a", 345216),
    "VizWiz_val_00000587": ("7a61bff023075d07603cba5a45fb04496d618637f06661493006d92d42a6e4a2", 457953),
    "VizWiz_val_00000588": ("0f2c522b72c1e9761b08f6ab198c36e6954ee2f289ec243614c9c175e5ef3f93", 708685),
    "VizWiz_val_00000592": ("14b8ad636f52b3ae4ec0c1558b642bf70176964a3d1abe052efb8fd186e1bfd5", 400201),
    "VizWiz_val_00000593": ("5a96a3263b6f822753a475f7decef1d1ba927fe12b04dc916ff85957df5710fd", 448192),
    "VizWiz_val_00000600": ("2240f4106911e8e003c0236cc5a1c66a62903a7e0539322bae98367b7d3eff41", 608205),
    "VizWiz_val_00000605": ("b91a665b834b790cd83ec9ca6b3116ee3d068eb36ce7ecf2e701387d0738b78e", 439179),
    "VizWiz_val_00000609": ("ae2c8739f4d7a434060c603961c46ec51e24883a1344380a61daf96f5c65cedd", 92203),
    "VizWiz_val_00000612": ("d9785493bfe988fff53b1109cd78b36705932c7600ba019f1c8abc67673c0d2f", 380167),
    "VizWiz_val_00000615": ("189c1f43d985bbfb8435a25f88b32e5a43f88c9fa1b76722be51c6c4b690f1a7", 553214),
    "VizWiz_val_00000619": ("0fc603fb6c625d9aea87b5ccc0ed87995ff9747d60a1953d3de5afe4d0c1bab0", 47898),
    "VizWiz_val_00000620": ("92ee8c61b4b76807ddf34be0c851715025dcd6cba4320b249e24aba19d963691", 526864),
    "VizWiz_val_00000622": ("a9f11463a4065d7b40be2bf92ef0b7430961843bc7803fb329ba9a4f0999f1a1", 644904),
    "VizWiz_val_00000624": ("b624b547ce4a40b4f3286f4bc2ec4bc9965891139e55c893270fcaf7fcfa1f9c", 641280),
    "VizWiz_val_00000627": ("53eb6ed7c2e8aaae97d783e756df2388a5e35ac05335f82fdcd0c174f7c4a693", 233542),
    "VizWiz_val_00000628": ("79e346543a5390eff18bea3ab563283bcafe953777cb56b9b5149864536f3e55", 232635),
    "VizWiz_val_00000629": ("5ca7273d91c1130128f27d5c37a1b15e47f83171c4e08ada3a661f1e9fc62a2c", 293241),
    "VizWiz_val_00000631": ("f03e1593cffec840103799fe9b3d1f6d138f493be34e68e4c4cf62b4b633c1ca", 447621),
    "VizWiz_val_00000634": ("97f256faad76f224faf3e4bd583f3be853d472275536c665a108dec5eff28b76", 119203),
    "VizWiz_val_00000638": ("7f17ad9527264310d77c438154ce8ec513c4760fa4b2e68135f1fd05827c593b", 41654),
    "VizWiz_val_00000639": ("2c8c5499da3368ac0328b31df545bfaee9d83a00648ac14591e400b8d1c6058a", 60705),
    "VizWiz_val_00000641": ("cc0792682c5b80292089b37f48cfd29e98af179931563c9653580dd1e5e5f747", 564991),
    "VizWiz_val_00000645": ("38f7611ad82922e3a78ebab8026543897c7bb32b327475cc57827c38099fe86b", 686440),
    "VizWiz_val_00000649": ("8be069ac59c47fa2363273cbd11799bc69dff7150ef9d0bf2a2209107ede16b6", 651667),
    "VizWiz_val_00000657": ("e54bafcd8ca1b756ece79560b55aced1df0507593b34484b2ec87519a3944ca9", 418109),
    "VizWiz_val_00000660": ("c6622dc48dc6ce3f6d240acc6a5bda8e42f0eb7f243cd3dd8ef0f9a3f1b1227e", 59668),
    "VizWiz_val_00000666": ("03f02b6f0f4a574a1febed32b40615b038afc6c430841f3ac773d14c6b499b17", 292916),
    "VizWiz_val_00000670": ("2dcc964f606ab42149db1ccb760e035587a1182e41cc9fb1ab4e107a4622ef56", 518496),
    "VizWiz_val_00000676": ("00493399e17cafa63cd55c901f2f8a80fb69935610a42396647e32d98f09fa87", 280359),
    "VizWiz_val_00000679": ("78b89cb31f8ea6850186043b32ae1e846d47b564ac48c5cec85a79ec19a91aa6", 69973),
    "VizWiz_val_00000681": ("07b20b19bc3561c241c58c43c00039918305a2b9434d0b141bbe54905d8f74b7", 283018),
    "VizWiz_val_00000684": ("5c86ffe915b8b76fa7c818c8857170a31c5cd36af02d6915be76b33ef70a26f2", 183812),
    "VizWiz_val_00000685": ("ad0a7ad352848ae2fa2a751b4caf117c2349c535f39c7ba11816d5be9f77d8f2", 305166),
    "VizWiz_val_00000689": ("bd362e03dec66a45f1f0a940fa56c97864062ef23d94c8320744e9801928a3e8", 358201),
    "VizWiz_val_00000690": ("2b33babfc58261506190525776e081fd8a253e6652c0a5c4667a43c162fd297e", 606926),
    "VizWiz_val_00000691": ("788a39c073b262599990f92e41f008119c99b24ad4f380909f72b210803673cd", 411789),
    "VizWiz_val_00000694": ("61fec6ff9872633fd8833bb00fb3bd040fb865e2ad5a3912014838de794597f1", 589026),
    "VizWiz_val_00000700": ("0ff6a89fec0937a274eee75f61a1602864db3d7963cae7acedf31b9ad6673757", 670973),
    "VizWiz_val_00000702": ("a2600eb99d04308075bf1d0ae9382820efa526f8f37539a3dc5d3809b4cc02c8", 410524),
    "VizWiz_val_00000703": ("fbf69d1ed3e3e22795ebbf86effcf4d24dc1f772d5b05b7ce8f0f4256e44e991", 336291),
    "VizWiz_val_00000705": ("3e776fd2ad36357bb7fd341a2a78525eee4d6d56c94fd9a1d1a35baa587af5eb", 577437),
    "VizWiz_val_00000709": ("1916553dc141ed3dac4ca3d34391d4b5550f9553ef4485479cf23fdf7a497255", 319851),
    "VizWiz_val_00000715": ("85db27edfab3ca500ce948bdd07ef8d45b6a05d990f65f78578d055d1b7dce48", 542650),
    "VizWiz_val_00000719": ("fe41e99ca2b7c14e5bc0110c9e989260db6e4c20a6d5ae5052271925ae91ae9e", 526101),
    "VizWiz_val_00000720": ("f600166d418dc914b42f072576d987d89a6bf308e41a7fe21c5510bb6bd2cd01", 516992),
    "VizWiz_val_00000725": ("634d55fbab1b8c6c3833bef00114be5279e6bf489ff8362c5e8f25a1836bfa18", 49234),
    "VizWiz_val_00000732": ("3dcfb27d44ad7dd97d627ee69905249b2772dc3f7364fa00c425cedf9758aa40", 561771),
    "VizWiz_val_00000740": ("775d5238e01bab8c886e684ad6c653131489d53f900b83b4dc7976341bff77e8", 447510),
    "VizWiz_val_00000741": ("9adbfe44270ebc01a3137a53bfa67197551201f0716b2ece73f6b3a3ff98b1e3", 580636),
    "VizWiz_val_00000743": ("d3f67ee19ac526601768cd96b489afe979e4c07901535d3ec5f057fbb4768adc", 584795),
    "VizWiz_val_00000755": ("1f9fcf8ff773446a64cc617a02b3eb638d4d37e5295959d1814b09947ba9b291", 322054),
    "VizWiz_val_00000757": ("a1c57385f9044ee2246e3193d17b21d6e834895d6dd8dd161269f9639c5ff266", 433365),
    "VizWiz_val_00000760": ("595e28a5153311019d3c933073feab1dea41aedb7a994f7b0a31f77a206d3dee", 169619),
    "VizWiz_val_00000768": ("e602575ddbecad8636f0905f6fca5d7608a3c8e309711962bec6676c01631b9f", 56284),
    "VizWiz_val_00000771": ("77e1bcfd1a7e197d476e20b1fc4efa65130fe53285341c90721f7218049649c3", 420021),
    "VizWiz_val_00000776": ("64ba907895f499199caa7b6195dfffad4a2921b233260645625de5f0350bc49d", 33732),
    "VizWiz_val_00000779": ("4f5180e63084bb49fb0196bae16b7fedfb9d4c199143b1eb6f797199fa53834e", 456232),
    "VizWiz_val_00000780": ("0967a445f0761a3d8e1541a1d5a70aabc1e9a016a48051495fe0ae65ee01b6e6", 394853),
    "VizWiz_val_00000783": ("3163f16379195d98447af5b9347c38674482b31616c175ea5f9c9b258e39474d", 297234),
    "VizWiz_val_00000784": ("82b90985d2319c5b18e80d7a1ef20a9dc17926dd38cecff18c028002be15eeac", 524191),
    "VizWiz_val_00000785": ("0b337fa9d1f8c617bd3f2ac1944abc3f1f3ac9ec0ff4814c8d16ad493d095a89", 332778),
    "VizWiz_val_00000791": ("4fdcca6010eea0749ecc6d9b0c9311f0f91ffe909c539f901db8d126af9d41ba", 474701),
    "VizWiz_val_00000792": ("8a14dd25bf7f1b9daaf0f4b90fb0fe75b83a6ddb7b8fda7066e8ab5345f5f6fa", 291803),
    "VizWiz_val_00000798": ("0508f59b10d56f6ef7d21db17fb2c196215d326d94815184a78264fc03cee259", 347680),
    "VizWiz_val_00000802": ("e2620c80196355a06ce36249386f4a1edac80ea4087584b2ca8d3b20a1eaa1c1", 1074885),
    "VizWiz_val_00000805": ("631dc871783ab81d32e7f89e9d74542a3a5a0b3c4b56c216bdcdc168deccc6ec", 367607),
    "VizWiz_val_00000806": ("aef0799edbe888718dbf6cf8e0cd8c5b56744bc4a887d3981870e83d96fe2539", 364172),
    "VizWiz_val_00000807": ("c119282caf5ac92a6569b276519401511bb82020f7e9f361c8b5b04d2835dd23", 320038),
    "VizWiz_val_00000810": ("2a55b7ce8685bed54a5eaa14686102de3825a265aa6a5b49841ac49c603208b8", 680556),
    "VizWiz_val_00000814": ("879f11d25c8a8d41b04490dbd663549a51b75fc12776be29cceb6415620dd36e", 421122),
    "VizWiz_val_00000816": ("c9962c1d1a66fed3b2e78382e1df3690e3744fb11a6412edfd1f1e28221723b9", 559996),
    "VizWiz_val_00000819": ("aef0799edbe888718dbf6cf8e0cd8c5b56744bc4a887d3981870e83d96fe2539", 364172),
    "VizWiz_val_00000826": ("6c3b0e0e0134a49fd36c59aded31a03c4614c142e500ecf1d832fd6f7a2f7f83", 695054),
    "VizWiz_val_00000827": ("cc7f2808da122a386d29597d189ec20c7cd3adc9b3aa66a948501da5cbdcdfae", 359437),
    "VizWiz_val_00000830": ("64a5fe060201093cb7b5d6e09f4ad045b5cf6710eca25b37cfdb7180eac42240", 209578),
    "VizWiz_val_00000836": ("26a19acc9c514d922e2d719623a58fb86c8e7904dd42680a0bfcd01d6eb06269", 344109),
    "VizWiz_val_00000840": ("7f3a18430adc3c38d3004186da9aa74f3649518f4acf56a79882869a42887403", 757636),
    "VizWiz_val_00000842": ("c99587d3e4d979f0c2498530ee9f4e9bc0a5ddedc5a1725ff429af83b9f556e6", 563695),
    "VizWiz_val_00000846": ("3533dc25035147eb4ee8e251e466e7708897f9b1b623e6c484231be540b70a16", 423756),
    "VizWiz_val_00000859": ("c8ce4b04a9db5441e81fdd87ac4cfe385d2ee010fd9f8ada365ea8ab3e870511", 354458),
    "VizWiz_val_00000860": ("f6baddec9ab81f34ccc64a082199b2b625016b7978eb1ff28fbb74ce54aa6502", 582255),
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def column_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """SHA-256 of the decoded text columns as a canonical JSON list of `[question_id, question, answers,
    category]`."""
    payload = [
        [str(r["question_id"]), str(r["question"]), [str(a) for a in r["answers"]], str(r["category"])]
        for r in rows
    ]
    return _sha256_bytes(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


class _HttpRangeFile(io.RawIOBase):
    """A seekable read-only view of one HTTPS object served with `Range` requests (what `pyarrow` needs to
    read a parquet footer and a few column chunks without downloading the file)."""

    def __init__(self, url: str, size: int) -> None:
        self.url, self.size, self.pos = url, size, 0
        self.fetched = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        if n <= 0 or self.pos >= self.size:
            return b""
        end = min(self.size, self.pos + n) - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310 (pinned https URL)
            if response.status != 206:
                raise ValueError(f"{self.url}: server ignored the Range request (HTTP {response.status})")
            data = response.read()
        self.fetched += len(data)
        self.pos += len(data)
        return data

    def readinto(self, buffer: Any) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def _hub_columns() -> list[dict[str, Any]]:
    """Read the pinned shard's text columns from the Hub: the file's declared size and LFS SHA-256 are checked
    against the pins first, then only the parquet footer and those columns are fetched."""
    import pyarrow.parquet as pq
    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    url = hf_hub_url(CORPUS_REPO, CORPUS_FILE["path"], repo_type="dataset", revision=CORPUS_REVISION)
    metadata = get_hf_file_metadata(url)
    declared = (metadata.etag or "").strip('"')
    if metadata.size != CORPUS_FILE["bytes"] or declared != CORPUS_FILE["sha256"]:
        raise ValueError(
            f"annotation shard: the Hub declares {metadata.size} bytes / sha256 {declared[:16]}…, "
            f"pinned {CORPUS_FILE['bytes']} / {CORPUS_FILE['sha256'][:16]}…"
        )
    table = pq.ParquetFile(_HttpRangeFile(url, CORPUS_FILE["bytes"])).read(columns=list(CORPUS_COLUMNS))
    return table.to_pylist()


def fetch_annotations(
    *, cache_dir: str | Path | None = None, fetcher: Callable[[], Sequence[Mapping[str, Any]]] | None = None
) -> list[dict[str, Any]]:
    """Return the shard's 864 annotation rows from the cache or the Hub, digest-verified."""
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / "annotations.json"
    rows: list[dict[str, Any]] | None = None
    if local.is_file():
        cached = json.loads(local.read_text(encoding="utf-8"))
        if isinstance(cached, list) and column_digest(cached) == CORPUS_FILE["column_sha256"]:
            rows = cached
    if rows is None:
        raw = fetcher() if fetcher is not None else _hub_columns()
        rows = [
            {
                "question_id": str(r["question_id"]),
                "question": str(r["question"]),
                "answers": [str(a) for a in r["answers"]],
                "category": str(r["category"]),
            }
            for r in raw
        ]
        if len(rows) != CORPUS_FILE["rows"] or column_digest(rows) != CORPUS_FILE["column_sha256"]:
            raise ValueError(
                f"annotation shard: fetched {len(rows)} rows with column sha256 {column_digest(rows)[:16]}…, "
                f"pinned {CORPUS_FILE['rows']} / {CORPUS_FILE['column_sha256'][:16]}…"
            )
        local.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def _fetch_with_backoff(url: str, *, sleep: Callable[[float], None] = time.sleep) -> bytes:
    """GET one pinned object; retry with exponential backoff (honouring a numeric Retry-After) on the
    transient
    statuses in `FETCH_RETRY_STATUSES`; any other failure propagates."""
    delay = 1.0
    for attempt in range(1, FETCH_MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 (pinned https URL)
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in FETCH_RETRY_STATUSES or attempt == FETCH_MAX_ATTEMPTS:
                raise
            retry_after = exc.headers.get("Retry-After", "") if exc.headers else ""
            wait = float(retry_after) if re.fullmatch(r"\d+(\.\d+)?", retry_after or "") else delay
            sleep(min(wait, 60.0))
            delay = min(delay * 2, 60.0)
    raise RuntimeError("unreachable")


def fetch_images(
    question_ids: Sequence[str],
    *,
    cache_dir: str | Path | None = None,
    fetcher: Callable[[str], bytes] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Path]:
    """Stage the pinned image of every question id into the cache, refusing any byte-size or SHA-256 mismatch
    (a cached file that no longer matches is re-fetched once). Returns question id -> image path."""
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    (cache / "images").mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    fetched = 0
    for qid in question_ids:
        if qid not in IMAGE_PINS:
            raise ValueError(f"{qid} is not one of the {len(IMAGE_PINS)} pinned sample images")
        digest, size = IMAGE_PINS[qid]
        dest = cache / "images" / f"{qid}.jpg"
        data = dest.read_bytes() if dest.is_file() else None
        if data is None or len(data) != size or _sha256_bytes(data) != digest:
            if fetched:
                sleep(FETCH_SPACING_SECONDS)
            data = (
                fetcher(qid)
                if fetcher is not None
                else _fetch_with_backoff(f"{IMAGE_HOST}{qid}.jpg", sleep=sleep)
            )
            fetched += 1
            if len(data) != size or _sha256_bytes(data) != digest:
                raise ValueError(
                    f"{qid}.jpg: fetched {len(data)} bytes with sha256 {_sha256_bytes(data)[:16]}…, "
                    f"pinned {size} / {digest[:16]}…"
                )
            dest.write_bytes(data)
        out[qid] = dest
    return out


def build_sample_dataset(
    annotations: Sequence[Mapping[str, Any]],
    *,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
    image_paths: Mapping[str, Path] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """The pinned 240-question sample (every question whose image is pinned), shuffled with `seed` and cut
    **by image** into `sizes`; `image_paths` (from `fetch_images`) fills each record's `image`."""
    sizes = dict(sizes or SAMPLE_SPLIT)
    by_id = {str(r["question_id"]): r for r in annotations}
    missing = [qid for qid in IMAGE_PINS if qid not in by_id]
    if missing:
        raise ValueError(
            f"{len(missing)} pinned question id(s) are absent from the annotations: {missing[:3]}"
        )
    pool = sorted(IMAGE_PINS)
    if sum(sizes.values()) > len(pool):
        raise ValueError(f"{len(pool)} pinned questions; the split sizes need {sum(sizes.values())}")
    random.Random(seed).shuffle(pool)
    out: dict[str, list[dict[str, Any]]] = {}
    offset = 0
    for name in ("train", "validation", "test"):
        chosen = pool[offset : offset + sizes[name]]
        offset += sizes[name]
        out[name] = [
            {
                "id": f"{name}-{index:04d}",
                "image_id": qid,
                "question": by_id[qid]["question"],
                "answers": [str(a) for a in by_id[qid]["answers"]],
                "category": str(by_id[qid]["category"]),
                "image": str(image_paths[qid])
                if image_paths is not None and qid in image_paths
                else f"{qid}.jpg",
            }
            for index, qid in enumerate(chosen)
        ]
    return out


def fetch_sample_dataset(
    *,
    cache_dir: str | Path | None = None,
    annotation_fetcher: Callable[[], Sequence[Mapping[str, Any]]] | None = None,
    image_fetcher: Callable[[str], bytes] | None = None,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """The tutorial splits from the pinned annotations and images."""
    annotations = fetch_annotations(cache_dir=cache_dir, fetcher=annotation_fetcher)
    paths = fetch_images(sorted(IMAGE_PINS), cache_dir=cache_dir, fetcher=image_fetcher)
    return build_sample_dataset(annotations, seed=seed, sizes=sizes, image_paths=paths)


def _check_record(record: Any, index: int, *, base_dir: Path | None) -> dict[str, Any]:
    label = f"records[{index}]"
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} must be a mapping with id/image/question/answers")
    for key in ("id", "image", "question", "answers"):
        if key not in record:
            raise ValueError(f"{label} is missing {key!r}")
    rid = record["id"]
    if not isinstance(rid, str) or not _ID_RE.match(rid):
        raise ValueError(f"{label}: id must match {_ID_RE.pattern}")
    image_ref = record["image"]
    if not isinstance(image_ref, (str, Path)) or not str(image_ref).strip():
        raise ValueError(f"{label}: image must be a file path")
    path = Path(image_ref)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    if not path.is_file():
        raise ValueError(f"{label}: image file not found: {path}")
    try:
        with Image.open(path) as handle:
            handle.load()
            _rgb, question, _tokens = _check_inputs(handle, record["question"], 1)
            width, height = handle.size
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"{label}: image cannot be decoded: {exc}") from exc
    answers = record["answers"]
    if isinstance(answers, str) or not isinstance(answers, Sequence) or len(answers) < MIN_ANSWERS:
        raise ValueError(f"{label}: answers must be a list of at least {MIN_ANSWERS} accepted answer(s)")
    checked_answers = [" ".join(str(a).split()) for a in answers]
    if not all(checked_answers):
        raise ValueError(f"{label}: every accepted answer must be a non-empty string")
    item = {
        "id": rid,
        "image_id": str(record.get("image_id", rid)),
        "question": question,
        "answers": checked_answers,
        "category": str(record.get("category", "other")),
        "image": str(path),
        "image_size": [width, height],
    }
    return item


def validate_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    min_records: int = MIN_RECORDS,
    max_records: int = MAX_RECORDS,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Structural validation of a VQA dataset (every image opened and decoded); raises ValueError before any
    model import."""
    if isinstance(records, Mapping) or not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("records must be a list of {id, image, question, answers} mappings")
    if not min_records <= len(records) <= max_records:
        raise ValueError(f"{len(records)} records; {min_records}..{max_records} are required")
    base = Path(base_dir) if base_dir is not None else None
    checked = []
    ids: set[str] = set()
    images: set[str] = set()
    for index, record in enumerate(records):
        item = _check_record(record, index, base_dir=base)
        if item["id"] in ids:
            raise ValueError(f"duplicate id {item['id']!r}")
        ids.add(item["id"])
        images.add(item["image_id"])
        checked.append(item)
    return {
        "records": checked,
        "n_records": len(checked),
        "unique_images": len(images),
        "categories": dict(Counter(r["category"] for r in checked)),
        "answers_per_question": {
            "min": min(len(r["answers"]) for r in checked),
            "max": max(len(r["answers"]) for r in checked),
        },
        "question_chars": {
            "min": min(len(r["question"]) for r in checked),
            "max": max(len(r["question"]) for r in checked),
        },
        "max_question_chars": MAX_QUESTION_CHARS,
        "digest": dataset_digest(checked),
        "model_id": MODEL_ID,
    }


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        [r["id"], r["image_id"], r["question"], list(r["answers"]), r.get("category", "")] for r in records
    ]
    return _sha256_bytes(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def gold_texts(record: Mapping[str, Any]) -> list[str]:
    """The accepted answers of a record."""
    return [str(a) for a in record["answers"]]


def check_split_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Assert no image id appears in two splits (leakage check)."""
    seen: dict[str, str] = {}
    for name, records in splits.items():
        for record in records:
            key = str(record.get("image_id", record["id"]))
            if key in seen and seen[key] != name:
                raise ValueError(f"image {key!r} appears in both {seen[key]} and {name}")
            seen[key] = name
    return {name: len(records) for name, records in splits.items()}


def split_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.2,
    seed: int = 0,
    base_dir: str | Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Seeded split of a BYOD dataset into train/validation/test **by image**: every question on the same
    image
    lands in the same split, so a test image is never seen in training."""
    if not (0.0 <= val_fraction < 1.0 and 0.0 < test_fraction < 1.0 and val_fraction + test_fraction < 1.0):
        raise ValueError("fractions must satisfy 0 <= val < 1, 0 < test < 1, val + test < 1")
    checked = validate_dataset(records, base_dir=base_dir)["records"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in checked:
        groups.setdefault(record["image_id"], []).append(record)
    order = list(groups.values())
    random.Random(seed).shuffle(order)
    n_test = max(1, round(len(checked) * test_fraction))
    n_val = round(len(checked) * val_fraction)
    splits: dict[str, list[dict[str, Any]]] = {"test": [], "validation": [], "train": []}
    for group in order:
        if len(splits["test"]) < n_test:
            splits["test"].extend(group)
        elif len(splits["validation"]) < n_val:
            splits["validation"].extend(group)
        else:
            splits["train"].extend(group)
    if len(splits["train"]) < MIN_RECORDS:
        raise ValueError(
            f"split leaves {len(splits['train'])} training records; at least {MIN_RECORDS} are required"
        )
    return splits


def load_byod_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Read records from a JSON array or a JSONL file of ``{id, image, question, answers}`` objects; `image`
    paths are resolved relative to the file's directory by `validate_dataset(..., base_dir=...)`."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"dataset not found: {file_path}")
    suffix = file_path.suffix.lower()
    text = file_path.read_text(encoding="utf-8")
    if suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if suffix == ".json":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON dataset must be an array of records")
        return data
    raise ValueError("BYOD datasets must be .json or .jsonl")


def write_dataset_jsonl(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """One record per line in the shape `load_byod_dataset` reads back (image paths as given)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = ("id", "image_id", "image", "question", "answers", "category")
    with open(out, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps({k: record[k] for k in keys if k in record}, ensure_ascii=False) + "\n")
    return out
