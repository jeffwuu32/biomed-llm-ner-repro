# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: .venv (3.14.6.final.0)
#     language: python
#     name: python3
# ---

# %% [markdown] id="0"
# # Biomedical NER using LLMs
#
# This notebook serves to accomplish the following:
#
# - download raw data from a chosen dataset's public release;
# - load a published Gemma 4 adapter from Hugging Face;
# - reproduce reported metrics on the dataset's official test split.
#
# **Environment:** Google Colab, or any machine with a CUDA GPU.
#
# **Authentication:** None required. `HF_TOKEN` requests and auth warnings from Hugging Face may be safely ignored.

# %% id="1"
import json
import random
import re
from pathlib import Path
from collections import Counter

BASE_DIR = Path(".")

# %% [markdown] id="1a"
# ## GPU gate
#
# A GPU is required, and at least 12 GB of VRAM is recommended.

# %% id="1b"
# %pip install -q "transformers>=5.10.1" torch datasets pandas protobuf sentencepiece
# %pip install -q accelerate bitsandbytes "peft>=0.19.0" tabulate huggingface_hub tqdm

import torch

assert torch.version.cuda is not None, (
    "This is a CPU-only build of PyTorch (torch.version.cuda is None) -- a "
    "plain `pip install torch` grabbed the CPU wheel, common on Windows. "
    "Reinstall using the correct command from pytorch.org's install selector: "
    "https://pytorch.org/get-started/locally/"
)
assert torch.cuda.is_available(), (
    "No CUDA device visible to PyTorch -- this script needs a GPU throughout. "
    "In Colab: Runtime -> Change runtime type -> T4/L4 GPU. Elsewhere: confirm "
    "a GPU is attached (`nvidia-smi`) and drivers are installed."
)
_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
if _vram_gb < 12:
    print(f"WARNING: {_vram_gb:.1f} GB VRAM detected -- 12 GB is recommended "
          f"(see README). Lower EVAL_GEN_BATCH_SIZE (Configuration cell) if you hit an OOM.")

# %% [markdown] id="2"
# ## Configuration

# %% [markdown]
# ### Free variables

# %% id="3"
DATASET = "biored"    # "bc5cdr" | "biored"

# Required fields; see README "Adapter Releases" section
EVAL_ADAPTER_ID = None
HUB_BRANCH = None

EVAL_GEN_BATCH_SIZE = 16   # adjust based on available GPU RAM

# %%
assert EVAL_ADAPTER_ID and "/" in EVAL_ADAPTER_ID, (
    "EVAL_ADAPTER_ID must be set (above) to the published adapter's full 'hf_username/repo_name'."
)
assert HUB_BRANCH, (
    "HUB_BRANCH must be set (above) to the published adapter's tag/revision -- "
    "left as None, every Hub call below silently resolves to 'main' instead of "
    "the pinned release, defeating the README's Pinning guarantee."
)

# %% [markdown]
# ### Control variables
#
# Do not change casually, may affect results.

# %% id="3a"
SEED = 42

# None is a sentinel: resolved from the adapter's own pushed run_config.json
# below, rather than guessed -- see that cell for why.
ANCHOR_MAX_SECTION_CHARS = None

N_EVAL = None   # None = evaluate the entire test split -- the default, and what reported
                # numbers are computed on. Not a free knob like EVAL_GEN_BATCH_SIZE above:
                # capping it gives a real number back, just not a comparable one (use the
                # sanity-check cell below for a quick per-example look instead).

# Greedy decoding (do_sample=False) is the ONE fixed, reproducible operating
# point every model card's headline number is reported at -- not a sweep
# over temperatures. Leave this as-is unless you're deliberately exploring
# sampling variance, which won't match the published number.
GEN_DO_SAMPLE = False
GEN_TEMPERATURE = 0.3   # only applied when GEN_DO_SAMPLE=True
GEN_TOP_P = 0.95
GEN_TOP_K = 64
GEN_NO_REPEAT_NGRAM_SIZE = None   # leave None -- this format's anchor tokens legitimately repeat,
                                  # and n-gram blocking breaks them (see the generation-config cell)
GEN_STREAK_LIMIT = 8              # repeating-span-loop detection threshold (diagnostic + restart trigger)
GEN_MAX_PERIOD = 5
GEN_RESTART_ON_LOOP = True        # inference-time mitigation: on a detected loop, cut and regenerate
                                  # the remainder as a fresh mini-chunk. On by default -- see the
                                  # "Current mitigation" note near the loop-restart code below.
GEN_MAX_RESTARTS = 2
GEN_RESTART_ON_MAX_TOKENS = False

_ANCHOR_RNG = random.Random(SEED)   # drives assign_anchor_ids; drawn from sequentially, never re-seeded per call
# ^ its draws depend on this notebook's entire prior execution history --
# re-running any one cell that calls assign_anchor_ids (chunk-building,
# the truncation retry, or an eval-time restart) without re-running from
# this cell down changes every anchor id from that point on. Re-run top to
# bottom after a partial failure (e.g. an OOM), not just the cell that failed.

# %% [markdown] id="3b"
# ### Hardware assumption

# %% id="4"
# ---- Derived --------------------------------------------------------------
MAX_LENGTH          = 1024
ATTN_IMPLEMENTATION = "sdpa"

HAS_GPU = torch.cuda.is_available()   # always True past the GPU gate above; kept as an explicit flag for the helpers below
TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
print(f"GPU: {torch.cuda.get_device_name(0)} | dtype: {TORCH_DTYPE}")

import contextlib

@contextlib.contextmanager
def gpu_mem_block(name):
    """Prints peak VRAM used inside the block."""
    if not HAS_GPU:
        yield
        return
    torch.cuda.reset_peak_memory_stats()
    try:
        yield
    finally:
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"GPU memory [{name}]: peak={peak_gb:.2f} GB")

# %% id="5"
# Reserved <unusedN> vocab slots. Only the anchor/sep/close markers are
# fixed-position -- per-category type tokens (<unused3>, <unused4>, ...) are
# assigned per-dataset below (TYPE_TOKEN_BY_CATEGORY, from CATEGORIES).
ANCHOR_TOKEN       = "<unused0>"   # opens an input-side anchor marker: <unused0>{id}<unused1>
ANCHOR_CLOSE_TOKEN = "<unused1>"   # closes an input anchor marker
ANCHOR_SEP_TOKEN    = "<unused2>"  # per-span output separator: {id}<unused2>{text}{type_token}
FIRST_TYPE_TOKEN_ID = 3            # <unused3> is the first category-token slot

# %% [markdown] id="7a"
# ## Resolve base model
#
# `MODEL_ID` is read from the adapter's own `adapter_config.json`
# (`base_model_name_or_path`) rather than hand-set, so it can't drift out of
# sync with whatever adapter `EVAL_ADAPTER_ID` actually points at.

# %% id="7b"
from huggingface_hub import hf_hub_download

_adapter_cfg_path = hf_hub_download(
    repo_id=EVAL_ADAPTER_ID, filename="adapter_config.json", revision=HUB_BRANCH,
)
MODEL_ID = json.loads(Path(_adapter_cfg_path).read_text(encoding="utf-8"))["base_model_name_or_path"]

_model_size_match = re.fullmatch(r"google/gemma-4-(\w+)-it", MODEL_ID)
assert _model_size_match, (
    f"Adapter {EVAL_ADAPTER_ID!r} declares base model {MODEL_ID!r}, which "
    f"doesn't match the expected 'google/gemma-4-<SIZE>-it' pattern -- this "
    f"notebook doesn't know how to load it."
)
MODEL_SIZE = _model_size_match.group(1)   # e.g. "E2B" | "E4B"

# Pinned per-size, not per-MODEL_ID-in-general -- each size is a separate Hub
# repo with its own commit history, so a SHA verified for one is meaningless
# for another. Only E2B is pinned (the only size either published adapter
# uses); an unlisted size falls back to unpinned rather than silently
# reusing the wrong commit.
_BASE_MODEL_REVISIONS = {
    "E2B": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",  # google/gemma-4-E2B-it @ main, 2026-07-20
}
BASE_MODEL_REVISION = _BASE_MODEL_REVISIONS.get(MODEL_SIZE)
if BASE_MODEL_REVISION is None:
    print(f"WARNING: no pinned revision known for MODEL_SIZE={MODEL_SIZE!r} -- "
          f"loading {MODEL_ID} at its current main HEAD, unpinned.")

print(f"Model: {MODEL_ID} (resolved from {EVAL_ADAPTER_ID!r})")

# %% [markdown] id="8"
# ## Validate run config
#
# Fetched unconditionally (not just when `ANCHOR_MAX_SECTION_CHARS` is left
# `None`) because it's also how `DATASET` gets cross-checked against the
# adapter: `EVAL_ADAPTER_ID` pointing at a real adapter trained on the OTHER
# dataset would otherwise run to completion and produce a file full of
# meaningless numbers, with no error. `ANCHOR_MAX_SECTION_CHARS` itself
# shapes the exact input-token distribution the chunking below builds -- to
# reproduce the adapter's real numbers this must match what it was actually
# trained with, not a locally-guessed value.

# %% id="9"
_run_cfg_path, _hf_error = None, None
try:
    _run_cfg_path = hf_hub_download(
        repo_id=EVAL_ADAPTER_ID, filename="run_config.json", revision=HUB_BRANCH,
    )
except Exception as e:
    _hf_error = e
assert _run_cfg_path is not None, (
    f"run_config.json not found on {EVAL_ADAPTER_ID!r} (revision={HUB_BRANCH!r}) "
    f"({_hf_error!r}). If it's genuinely missing from this adapter, set "
    f"ANCHOR_MAX_SECTION_CHARS explicitly in the Configuration cell instead "
    f"(see that adapter's model card) -- but note DATASET can no longer be "
    f"cross-checked against the adapter without this file."
)
_run_cfg = json.loads(Path(_run_cfg_path).read_text(encoding="utf-8"))

_run_cfg_dataset = _run_cfg.get("dataset")   # absent on older run_config.json -- skip, don't fail
if _run_cfg_dataset is not None:
    assert _run_cfg_dataset == DATASET, (
        f"DATASET={DATASET!r} (Configuration cell) doesn't match "
        f"{EVAL_ADAPTER_ID!r}'s own run_config.json (dataset={_run_cfg_dataset!r}) "
        f"-- this adapter was trained on a different dataset than the one selected."
    )

if ANCHOR_MAX_SECTION_CHARS is None:
    ANCHOR_MAX_SECTION_CHARS = _run_cfg["anchor_max_section_chars"]
    print(f"ANCHOR_MAX_SECTION_CHARS resolved from {EVAL_ADAPTER_ID!r}: {ANCHOR_MAX_SECTION_CHARS}")
else:
    print(f"ANCHOR_MAX_SECTION_CHARS={ANCHOR_MAX_SECTION_CHARS} (from the Configuration cell).")

# %% [markdown] id="10"
# ## Download & load data
#
# Downloads the raw corpus directly from its public release (no DUA, no
# account, nothing to upload) and parses the shared PubTator format
# in-memory.

# %% id="11"
import hashlib
import urllib.request
import zipfile
from dataclasses import dataclass, field

import pandas as pd

# BC5CDR is pinned to a commit (not `master` -- the original repo went down
# and this community mirror is a live branch, so an un-pinned URL could
# silently start serving different content). BioRED's URL is already a
# static, dated NCBI FTP artifact, not a branch tip, so it doesn't need the
# same treatment. Both get a SHA256 check below regardless, since neither
# host publishes one of its own to verify against.
_DATASET_INFO = {
    "bc5cdr": {
        "categories": ["Chemical", "Disease"],
        "url": "https://raw.githubusercontent.com/JHnlp/BioCreative-V-CDR-Corpus/dd0b3165a94338874979a2a8aaedf74b67466b9c/CDR_Data.zip",
        "split_files": {
            "train": "CDR_Data/CDR.Corpus.v010516/CDR_TrainingSet.PubTator.txt",
            "dev":   "CDR_Data/CDR.Corpus.v010516/CDR_DevelopmentSet.PubTator.txt",
            "test":  "CDR_Data/CDR.Corpus.v010516/CDR_TestSet.PubTator.txt",
        },
    },
    "biored": {
        "categories": ["ChemicalEntity", "DiseaseOrPhenotypicFeature", "GeneOrGeneProduct",
                       "OrganismTaxon", "SequenceVariant", "CellLine"],
        "url": "https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/BIORED.zip",
        "split_files": {
            "train": "BioRED/Train.PubTator",
            "dev":   "BioRED/Dev.PubTator",
            "test":  "BioRED/Test.PubTator",
        },
    },
}

# Neither host publishes its own checksum -- these were computed once (from
# the URLs above, verified against a byte-for-byte-identical `master` fetch
# for bc5cdr) and hardcoded, so a future silent change in either corpus's
# contents fails loudly instead of quietly changing what gets reproduced.
_CORPUS_SHA256 = {
    "bc5cdr": "0a359a7f038d283a7b05b084fa73de014e7410e1f5d7034bf3fd01f016fc2444",
    "biored": "c3032230bd89d22a0923d0df6ae943bc8ea37fba7e42dafa7a8dec21bac02d47",
}
assert DATASET in _DATASET_INFO, f"Unknown DATASET {DATASET!r}; choose {sorted(_DATASET_INFO)}."
_info = _DATASET_INFO[DATASET]
CATEGORIES = _info["categories"]
CHUNKING_STRATEGY = "fixed_window"   # both corpora are PubMed abstracts -- no section-header structure

TYPE_TOKEN_BY_CATEGORY = {
    cat: f"<unused{FIRST_TYPE_TOKEN_ID + i}>" for i, cat in enumerate(CATEGORIES)
}
print(f"Dataset: {DATASET!r} | categories: {CATEGORIES}")

# %% id="11a"
# --- PubTator parser (BC5CDR/BioRED share this format) ----------------------
@dataclass
class PubTatorEntity:
    start: int
    end: int
    text: str
    type: str


@dataclass
class PubTatorDoc:
    pmid: str
    text: str   # title + " " + abstract
    entities: list = field(default_factory=list)


def _is_entity_line(fields):
    if len(fields) < 6:
        return False
    try:
        start, end = int(fields[1]), int(fields[2])
    except ValueError:
        return False
    return end > start


def parse_pubtator(path):
    """Yields one PubTatorDoc per document block (blocks separated by a
    blank line). Entity lines are identified structurally (6 tab-fields,
    integer offsets) -- anything else on a PMID-prefixed line is a relation
    line and is skipped (neither adapter needs relations)."""
    pmid = title = abstract = None
    entities = []

    def _flush():
        if pmid is None:
            return None
        return PubTatorDoc(pmid=pmid, text=(title or "") + " " + (abstract or ""), entities=entities)

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line.strip():
                doc = _flush()
                if doc is not None:
                    yield doc
                pmid, title, abstract, entities = None, None, None, []
                continue
            if "|t|" in line:
                pmid, _, title = line.partition("|t|")
                continue
            if "|a|" in line:
                pmid, _, abstract = line.partition("|a|")
                continue
            fields = line.split("\t")
            if _is_entity_line(fields):
                entities.append(PubTatorEntity(start=int(fields[1]), end=int(fields[2]),
                                                text=fields[3], type=fields[4]))
    doc = _flush()
    if doc is not None:
        yield doc

# %% id="11b"
# --- Download + build notes_df/annot_df --------------------------------------
_raw_dir = BASE_DIR / "raw" / DATASET
if not _raw_dir.exists():
    _zip_path = BASE_DIR / f"{DATASET}.zip"
    print(f"Downloading {_info['url']} ...")
    urllib.request.urlretrieve(_info["url"], _zip_path)
    _actual_sha256 = hashlib.sha256(_zip_path.read_bytes()).hexdigest()
    assert _actual_sha256 == _CORPUS_SHA256[DATASET], (
        f"{DATASET}.zip doesn't match the expected SHA256 "
        f"({_actual_sha256} != {_CORPUS_SHA256[DATASET]}) -- the corpus "
        f"changed upstream, or the download was corrupted. Not safe to "
        f"proceed without checking what changed."
    )
    print(f"Extracting to {_raw_dir} ...")
    with zipfile.ZipFile(_zip_path) as zf:
        zf.extractall(_raw_dir)
else:
    print(f"Reusing already-downloaded {_raw_dir}")

_note_rows, _annot_rows = [], []
for split, rel_path in _info["split_files"].items():
    path = _raw_dir / rel_path
    assert path.exists(), f"{path} not found after extraction -- corpus layout may have changed."
    for doc in parse_pubtator(path):
        _note_rows.append({"note_id": doc.pmid, "text": doc.text, "split": split})
        for ent in doc.entities:
            _annot_rows.append({"note_id": doc.pmid, "start": ent.start, "end": ent.end, "category": ent.type})

notes_df = pd.DataFrame(_note_rows)
annot_df = pd.DataFrame(_annot_rows)
print(f"Notes: {len(notes_df)} | Annotations: {len(annot_df)} | "
      f"split distribution: {notes_df['split'].value_counts().to_dict()}")

# %% [markdown] id="11c"
# ### Build the system prompt
#
# Mechanically generated from `CATEGORIES`/`TYPE_TOKEN_BY_CATEGORY` -- see
# `render_medium_prompt`'s docstring below for why this (not a hand-authored
# prompt) is what these adapters were actually trained against.

# %% id="11d"
def _join_or_and(items, conj):
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} {conj} {items[1]}"
    return ", ".join(items[:-1]) + f", {conj} {items[-1]}"


def render_medium_prompt(categories, type_token_by_category):
    """The only prompt variant this script supports -- mechanically
    generated from the category list, matching how these two adapters were
    actually trained (no dataset-specific hand-authored prompt exists for
    either)."""
    names = [c.replace("_", " ") for c in categories]
    cat_phrase = _join_or_and(names, "or")
    token_phrase = _join_or_and(
        [f"`{type_token_by_category[c]}` {n}" for c, n in zip(categories, names)], "or"
    )
    return (
        f"Extract text naming {cat_phrase} from the input. The text has "
        f"positional markers already embedded (`<unused0>N<unused1>`, e.g. "
        f"`<unused0>58<unused1>`) — each marker's id is arbitrary, not a "
        f"reading-order count, so markers do not necessarily appear in "
        f"ascending numeric order. These markers are not part of the content; "
        f"never extract one as a match.\n\n"
        f"Copy each match verbatim from the input, including any irregular "
        f"internal whitespace or embedded line breaks. Include negated, "
        f"uncertain, and hedged mentions, but not the negation/hedging words "
        f"themselves. Do not include anything outside the categories above.\n\n"
        f"Before every match, write the id of the nearest `<unused0>N<unused1>` "
        f"marker before it, followed by `<unused2>` — repeat this prefix before "
        f"EVERY match, even consecutive matches under the same marker; never "
        f"omit or dedupe it — then the match text, then its `<CATEGORY_TOKEN>`, "
        f"as `N<unused2>TEXT<CATEGORY_TOKEN>` (text first, category token last "
        f"— not the other way around, and no other characters — never write "
        f"markup like `<span>`). `<CATEGORY_TOKEN>` is one of: {token_phrase}. "
        f"Tag every occurrence separately, even repeats of the same text — do "
        f"not deduplicate. Output only this, with no explanation or markdown "
        f"fences."
    )


SYSTEM_PROMPT = render_medium_prompt(CATEGORIES, TYPE_TOKEN_BY_CATEGORY)
print(f"System prompt ({len(SYSTEM_PROMPT)} chars), copy-pasteable below:\n")
print("-" * 70)
print(SYSTEM_PROMPT)
print("-" * 70)

# %% [markdown] id="12"
# ## Chunking & anchor placement
#
# Notes are split into chunks (`CHUNKING_STRATEGY`, fixed to `"fixed_window"`
# for these two datasets); each chunk becomes a 3-message conversation with
# positional anchor markers embedded in the `user` message and gold spans in
# the `assistant` message. This is exactly how the adapter's own training
# data was built, not a reimplementation -- any deviation here would silently
# change what "reproducing" means.

# %% id="13"
note_text = dict(zip(notes_df["note_id"], notes_df["text"]))
note_split = dict(zip(notes_df["note_id"], notes_df["split"]))

_n_annot_before_typing = len(annot_df)
annot_df = annot_df.copy()
annot_df["type_token"] = annot_df["category"].map(TYPE_TOKEN_BY_CATEGORY)
_n_untyped = annot_df["type_token"].isna().sum()
annot_df = annot_df.dropna(subset=["type_token"])
print(f"Dropped {_n_untyped}/{_n_annot_before_typing} annotations with a "
      f"category outside this dataset's categories ({CATEGORIES}).")

HEADER_RE = re.compile(r"^[A-Z][A-Za-z /\-]{2,45}:\s*$", re.MULTILINE)

# Approximate char budget for chunk grouping (the tokenizer-based length
# check further below still drops any chunk that ends up over MAX_LENGTH
# regardless).
CHARS_PER_TOKEN_APPROX = 2.1
_FUDGE_TOKENS = 64
_prompt_tokens_approx = len(SYSTEM_PROMPT) / 3.2
CHUNK_CHAR_BUDGET = max(
    200, int((MAX_LENGTH - _prompt_tokens_approx - _FUDGE_TOKENS) * CHARS_PER_TOKEN_APPROX)
)
print(f"Approx per-chunk char budget: {CHUNK_CHAR_BUDGET} (MAX_LENGTH={MAX_LENGTH})")

# %% id="13a"
# ---- Chunking strategy: split a note's text into (start, end) spans -------
def _section_boundaries(text):
    starts = [m.start() for m in HEADER_RE.finditer(text)]
    bounds = sorted(set([0] + starts + [len(text)]))
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
            if bounds[i] < bounds[i + 1]]


def chunk_note_section_header(text, char_budget):
    sections = _section_boundaries(text)
    if not sections:
        return [(0, len(text))]
    chunks = []
    cur_start, cur_end = sections[0]
    for s, e in sections[1:]:
        if e - cur_start <= char_budget:
            cur_end = e
        else:
            chunks.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    chunks.append((cur_start, cur_end))
    return chunks


def chunk_note_fixed_window(text, char_budget):
    n = len(text)
    if n <= char_budget:
        return [(0, n)]
    chunks = []
    start = 0
    while start < n:
        end = min(start + char_budget, n)
        if end < n:
            snap = text.rfind(" ", max(start, end - 200), end)
            if snap > start:
                end = snap
        chunks.append((start, end))
        start = end
    return chunks


def chunk_note(text, char_budget):
    if CHUNKING_STRATEGY == "section_header":
        return chunk_note_section_header(text, char_budget)
    return chunk_note_fixed_window(text, char_budget)


def chunk_annotations(rows, chunk_start, chunk_end):
    return rows[(rows["start"] >= chunk_start) & (rows["end"] <= chunk_end)]

# %% [markdown] id="13b"
# ### Anchor placement
#
# Positional `<unusedN>` markers are placed at each chunk's section
# boundaries plus extra breakpoints every `ANCHOR_MAX_SECTION_CHARS`, nudged
# forward off any gold span so a marker never lands inside one. Each
# breakpoint gets a random id in `[0, 99]` (not sequential -- a training-time
# mitigation against the model latching onto positional id patterns).

# %% id="13c"
def _nudge_forward(pos, text, forbidden_ranges, max_search=300):
    n = len(text)
    for p in range(pos, min(pos + max_search, n + 1)):
        if p != 0 and p != n and not text[p - 1].isspace():
            continue
        if any(s <= p < e for s, e in forbidden_ranges):
            continue
        return p
    return pos


def compute_chunk_anchors(chunk_text, section_starts_local, forbidden_ranges):
    primary = sorted(set([0] + list(section_starts_local)))
    breakpoints = []
    for i, start in enumerate(primary):
        end = primary[i + 1] if i + 1 < len(primary) else len(chunk_text)
        breakpoints.append(start)
        run_len = end - start
        n_extra = run_len // ANCHOR_MAX_SECTION_CHARS
        for k in range(1, n_extra + 1):
            cand = start + k * ANCHOR_MAX_SECTION_CHARS
            if cand < end:
                breakpoints.append(cand)
    nudged = [0] + [
        _nudge_forward(bp, chunk_text, forbidden_ranges)
        for bp in breakpoints if bp > 0
    ]
    return sorted(set(nudged))


def assign_anchor(pos, breakpoints):
    idx = 0
    for i, bp in enumerate(breakpoints):
        if bp <= pos:
            idx = i
        else:
            break
    return idx + 1


def assign_anchor_ids(breakpoints, rng):
    assert len(breakpoints) <= 100, (
        f"{len(breakpoints)} breakpoints exceeds the [0,99] id pool."
    )
    return rng.sample(range(100), k=len(breakpoints))


def anchor_id_to_index(anchor_ids):
    return {aid: i + 1 for i, aid in enumerate(anchor_ids)}


_CATEGORY_BY_TYPE_TOKEN = {v: k for k, v in TYPE_TOKEN_BY_CATEGORY.items()}

# %% id="13d"
def _build_chunk_record(note_id, text, c_start, c_end, header_starts, note_rows):
    """One chunk record for note_id's [c_start:c_end) region. Reused as-is by
    the test-set length-truncation retry further down with a shrunk c_end --
    same construction either way, not a separate truncation code path."""
    chunk_text = text[c_start:c_end]
    kept = chunk_annotations(note_rows, c_start, c_end)

    local_ranges = [
        (int(r["start"]) - c_start, int(r["end"]) - c_start, r["type_token"])
        for _, r in kept.iterrows()
    ]
    section_starts_local = [s - c_start for s in header_starts if c_start <= s < c_end]
    _forbidden_ranges = [(s, e) for s, e, _ in local_ranges]
    anchors = compute_chunk_anchors(chunk_text, section_starts_local, _forbidden_ranges)
    anchor_ids = assign_anchor_ids(anchors, _ANCHOR_RNG)

    target_parts = []
    for l_start, l_end, type_token in local_ranges:
        anchor_idx = assign_anchor(l_start, anchors)
        display_id = anchor_ids[anchor_idx - 1]
        span_text = chunk_text[l_start:l_end]
        target_parts.append(f"{display_id}{ANCHOR_SEP_TOKEN}{span_text}{type_token}")
    target = "".join(target_parts)

    anchored_text = chunk_text
    for idx in range(len(anchors), 0, -1):
        pos = anchors[idx - 1]
        marker = f"{ANCHOR_TOKEN}{anchor_ids[idx - 1]}{ANCHOR_CLOSE_TOKEN}"
        anchored_text = anchored_text[:pos] + marker + anchored_text[pos:]

    gold_spans_local = [
        (l_start, l_end, _CATEGORY_BY_TYPE_TOKEN[type_token])
        for l_start, l_end, type_token in local_ranges
    ]
    record = {
        "note_id": note_id, "c_start": c_start, "c_end": c_end,
        "split": note_split[note_id],
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": anchored_text},
            {"role": "assistant", "content": target},
        ],
        "chunk_text": chunk_text,
        "anchors": anchors,
        "anchor_ids": anchor_ids,
        "gold_spans": json.dumps(gold_spans_local),
    }
    return record, kept

# %% id="13e"
# ---- Build every chunk record for every note --------------------------------
records = []
_n_notes = 0
_n_annotations_total = 0
_n_annotations_kept = 0
# Gold annotations that straddle a chunk boundary can't be represented whole
# in either resulting chunk, so they're excluded when building `records`
# above -- but for the TEST split, "excluded from the gold set" would let
# the reported P/R/F1 skip spans the method structurally can't reach,
# inflating it relative to literature numbers computed over the full
# official test set. Tracked here (test notes only, by category) so the
# Reproduce step below can charge them as misses instead -- both as a span
# count (category-match table) and total char length (positional table).
_test_boundary_drops_by_category = Counter()
_test_boundary_drop_chars_by_category = Counter()
for note_id in notes_df["note_id"]:
    text = note_text[note_id]
    if not isinstance(text, str) or not text:
        continue
    _n_notes += 1

    note_rows = annot_df[annot_df["note_id"] == note_id].sort_values("start")
    _n_annotations_total += len(note_rows)
    header_starts = ([m.start() for m in HEADER_RE.finditer(text)]
                      if CHUNKING_STRATEGY == "section_header" else [])

    _kept_idx_this_note = []
    for c_start, c_end in chunk_note(text, CHUNK_CHAR_BUDGET):
        record, kept = _build_chunk_record(note_id, text, c_start, c_end, header_starts, note_rows)
        _n_annotations_kept += len(kept)
        _kept_idx_this_note.extend(kept.index)
        records.append(record)

    if note_split[note_id] == "test":
        _dropped_this_note = note_rows.loc[~note_rows.index.isin(_kept_idx_this_note)]
        for _, _drow in _dropped_this_note.iterrows():
            _test_boundary_drops_by_category[_drow["category"]] += 1
            _test_boundary_drop_chars_by_category[_drow["category"]] += int(_drow["end"]) - int(_drow["start"])

_n_dropped = _n_annotations_total - _n_annotations_kept
print(f"Built {len(records)} chunks from {_n_notes} notes.")
if _n_dropped:
    print(f"Dropped {_n_dropped}/{_n_annotations_total} annotations "
          f"({100*_n_dropped/_n_annotations_total:.1f}%) that straddled a chunk boundary "
          f"({sum(_test_boundary_drops_by_category.values())} of those in the test split -- "
          f"charged as misses at eval time, not excluded).")

# %% [markdown] id="14"
# ## Load the processor & chat template
#
# The processor for `-E2B-it` ships the official Gemma 4 `chat_template.jinja`.

# %% id="15"
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(MODEL_ID, revision=BASE_MODEL_REVISION)
tokenizer = processor.tokenizer

assert getattr(processor, "chat_template", None) or getattr(tokenizer, "chat_template", None), \
    "No chat template found on the processor."
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Reserved <unusedN> slots don't split atomically by default even though
# they have a real vocab id -- add_tokens(special_tokens=False) registers
# atomic pre-tokenization, reusing the existing id (no resize).
_DELIM_TOKENS = (ANCHOR_TOKEN, ANCHOR_CLOSE_TOKEN, ANCHOR_SEP_TOKEN,
                  *TYPE_TOKEN_BY_CATEGORY.values())
_vocab_size_before = len(tokenizer)
_n_added = tokenizer.add_tokens(list(_DELIM_TOKENS), special_tokens=False)
assert len(tokenizer) == _vocab_size_before, (
    f"add_tokens grew the vocab by {_n_added} instead of reusing existing reserved ids."
)
for _tok in _DELIM_TOKENS:
    _ids = tokenizer.encode(_tok, add_special_tokens=False)
    assert len(_ids) == 1 and tokenizer.decode(_ids) == _tok, (
        f"{_tok!r} still isn't a single stable token after add_tokens (got {_ids})."
    )
    assert _ids[0] not in set(tokenizer.all_special_ids), (
        f"{_tok!r} landed in all_special_ids -- skip_special_tokens=True would strip it."
    )
print("Anchor/delimiter tokens OK:", {
    t: tokenizer.encode(t, add_special_tokens=False)[0] for t in _DELIM_TOKENS
})

_delim_ids = {t: tokenizer.encode(t, add_special_tokens=False)[0] for t in _DELIM_TOKENS}
_prompt_ids = tokenizer.encode(SYSTEM_PROMPT, add_special_tokens=False)
for _tok, _tid in _delim_ids.items():
    _expected = SYSTEM_PROMPT.count(_tok)
    _actual = _prompt_ids.count(_tid)
    assert _expected == _actual, (
        f"{_tok!r} appears {_expected}x as literal text but only {_actual}x as its "
        f"atomic token id in the encoded prompt -- fragmenting into multiple BPE pieces."
    )
print("Anchor/delimiter tokens verified atomic in-context.")

# %% [markdown] id="16"
# ## Isolate the official test split
#
# The whole point of this script: evaluate on notes that were never used for
# training or checkpoint selection.

# %% id="17"
from datasets import Dataset


def _to_flat_ids(x):
    if isinstance(x, dict):
        x = x["input_ids"]
    if torch.is_tensor(x):
        x = x.tolist()
    if len(x) > 0 and isinstance(x[0], (list, tuple)):
        x = x[0]
    return list(x)


def _token_len(messages):
    ids = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    return len(_to_flat_ids(ids))


# Only test-split rows are ever used by this script -- train/dev chunks
# aren't measured or truncated at all.
test_records = [r for r in records if r["split"] == "test"]
print(f"Test (before length check): {len(test_records)} chunks "
      f"({len(set(r['note_id'] for r in test_records))} notes).")

# %% id="17a"
# A chunk over MAX_LENGTH is re-chunked from the SAME note region down to a
# shorter span (via _build_chunk_record above -- not a separate truncation
# path) rather than dropped outright -- still evaluated, best-effort. Rare
# enough (<1% of chunks) that one directly-calculated shrink -- using this
# chunk's own measured token/char ratio, not a blind guess -- is enough; a
# chunk that still doesn't fit after that one retry is dropped, same as the
# boundary-straddling case above. Gold spans that fall outside the truncated
# region are tracked here (by category, and by total char length for the
# positional table) and charged as misses at eval time, not silently
# excluded.
_test_truncation_drops_by_category = Counter()
_test_truncation_drop_chars_by_category = Counter()
_n_truncated = 0
_n_dropped_after_truncation_retry = 0
final_test_records = []
for r in test_records:
    n_tokens = _token_len(r["messages"])
    if n_tokens <= MAX_LENGTH:
        final_test_records.append(r)
        continue

    note_id, text = r["note_id"], note_text[r["note_id"]]
    header_starts = ([m.start() for m in HEADER_RE.finditer(text)]
                      if CHUNKING_STRATEGY == "section_header" else [])
    note_rows = annot_df[annot_df["note_id"] == note_id].sort_values("start")
    c_start = r["c_start"]
    chars_per_token = (r["c_end"] - c_start) / n_tokens
    new_c_end = c_start + int(MAX_LENGTH * chars_per_token * 0.9)   # 0.9 safety margin

    fitted = None
    candidate, _kept = _build_chunk_record(note_id, text, c_start, new_c_end, header_starts, note_rows)
    if _token_len(candidate["messages"]) <= MAX_LENGTH:
        fitted = candidate

    if fitted is None:
        # One retry wasn't enough -- drop the whole chunk, same as a
        # boundary-straddling chunk, and charge ALL of its gold spans as
        # structural misses (not just the ones a partial truncation would
        # have lost) since none of them will ever be evaluated now.
        _n_dropped_after_truncation_retry += 1
        for l_start, l_end, cat in json.loads(r["gold_spans"]):
            _test_truncation_drops_by_category[cat] += 1
            _test_truncation_drop_chars_by_category[cat] += l_end - l_start
        continue

    _n_truncated += 1
    # c_start is unchanged by the retry (only c_end shrinks), so a surviving
    # span's local (start, end) is identical between the two attempts --
    # a plain multiset diff on the full tuple isolates exactly what was lost.
    original_spans = Counter(tuple(s) for s in json.loads(r["gold_spans"]))
    kept_spans = Counter(tuple(s) for s in json.loads(fitted["gold_spans"]))
    for (l_start, l_end, cat), _n_lost in (original_spans - kept_spans).items():
        _test_truncation_drops_by_category[cat] += _n_lost
        _test_truncation_drop_chars_by_category[cat] += (l_end - l_start) * _n_lost
    final_test_records.append(fitted)

if _n_truncated or _n_dropped_after_truncation_retry:
    print(f"Truncated {_n_truncated} test chunk(s) to fit MAX_LENGTH={MAX_LENGTH} (still "
          f"evaluated) and dropped {_n_dropped_after_truncation_retry} entirely (one retry "
          f"wasn't enough) -- {sum(_test_truncation_drops_by_category.values())} gold "
          f"annotation(s) total fell outside an evaluated region (charged as misses).")

test_ds = Dataset.from_list(final_test_records)
assert len(test_ds) > 0, "No test-split chunks survived -- check the downloaded corpus/MAX_LENGTH."
print(f"Test: {len(test_ds)} chunks ({len(set(test_ds['note_id']))} notes).")

# Structural misses (test split only): gold spans the method never had a
# chance to predict, from either drop path above. Charged as zero-credit
# misses on both the "Category match" and "Positional" tables further down --
# the former by span count, the latter by total char length -- not excluded.
# See the README's "What is reproduced" note.
TEST_STRUCTURAL_FN_BY_CATEGORY = _test_boundary_drops_by_category + _test_truncation_drops_by_category
TEST_STRUCTURAL_FN_CHARS_BY_CATEGORY = (
    _test_boundary_drop_chars_by_category + _test_truncation_drop_chars_by_category
)
if TEST_STRUCTURAL_FN_BY_CATEGORY:
    print(f"Structural misses to be charged at eval time: {dict(TEST_STRUCTURAL_FN_BY_CATEGORY)} "
          f"({dict(TEST_STRUCTURAL_FN_CHARS_BY_CATEGORY)} chars, for the positional table)")

# %% [markdown] id="18"
# ## Model load
#
# Loads the base model in 4-bit (GPU gate + installs already done at the
# top), then attaches the published LoRA adapter.

# %% id="20"
from transformers import AutoModelForMultimodalLM, BitsAndBytesConfig
from peft import PeftModel

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=TORCH_DTYPE,
    bnb_4bit_quant_storage=TORCH_DTYPE,
)
_model_kwargs = dict(quantization_config=bnb_config, dtype=TORCH_DTYPE, device_map="auto",
                     revision=BASE_MODEL_REVISION)
if ATTN_IMPLEMENTATION:
    _model_kwargs["attn_implementation"] = ATTN_IMPLEMENTATION

with gpu_mem_block("base model load"):
    model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, **_model_kwargs)

with gpu_mem_block("adapter load"):
    model = PeftModel.from_pretrained(model, EVAL_ADAPTER_ID, revision=HUB_BRANCH)
model.eval()
model.config.use_cache = True

# HUB_BRANCH is a tag, not a commit -- resolve it to the actual commit it
# points at right now, so saved provenance still identifies the exact
# adapter weights even if the tag were ever moved later.
from huggingface_hub import HfApi
_ADAPTER_COMMIT = HfApi().model_info(repo_id=EVAL_ADAPTER_ID, revision=HUB_BRANCH).sha

# A loaded all-zero LoRA B (untrained adapter) behaves exactly like the base
# model and would silently report base-model numbers as the adapter's.
_b_max = max((p.detach().abs().max().item()
              for n, p in model.named_parameters() if "lora_B" in n), default=0.0)
assert _b_max > 0, f"Loaded adapter {EVAL_ADAPTER_ID!r} has all-zero LoRA B weights -- untrained."
print(f"Loaded adapter {EVAL_ADAPTER_ID!r} (max|B|={_b_max:.4g}).")

# %% [markdown] id="21"
# ## Evaluation helpers
#
# Scored as **multisets** (exact-string match, true positives = the
# intersection): a **macro** (mean of the per-category P/R/F1, every category
# weighted equally) and a **micro** (pooled TP/FP/FN across all chunks, every
# span weighted equally) P/R/F1, plus the invalid-output rate. This is
# exactly the scoring code the published numbers were computed with, not a
# reimplementation.

# %% id="22"
from transformers import GenerationConfig
from tqdm.auto import tqdm

gen_cfg = GenerationConfig.from_pretrained(MODEL_ID, revision=BASE_MODEL_REVISION)
gen_cfg.max_new_tokens = 512
gen_cfg.do_sample = GEN_DO_SAMPLE
if GEN_DO_SAMPLE:
    gen_cfg.temperature = GEN_TEMPERATURE
    gen_cfg.top_p = GEN_TOP_P
    gen_cfg.top_k = GEN_TOP_K
if GEN_NO_REPEAT_NGRAM_SIZE:
    gen_cfg.no_repeat_ngram_size = GEN_NO_REPEAT_NGRAM_SIZE

# Stop at Gemma's end-of-turn token. In this tokenizer that token's literal
# string is "<turn|>" (id 106), NOT "<end_of_turn>" (which is UNK here).
_eot = tokenizer.convert_tokens_to_ids("<turn|>")
assert _eot is not None and _eot >= 0 and _eot != tokenizer.unk_token_id, (
    f"'<turn|>' did not resolve to a real token (got {_eot})."
)
gen_cfg.eos_token_id = [_eot]
print(f"eos stop token '<turn|>' -> id {_eot}.")


_TYPE_TOKEN_TO_CATEGORY = {v: k for k, v in TYPE_TOKEN_BY_CATEGORY.items()}
_TYPE_TOKEN_ALT = "|".join(re.escape(t) for t in _TYPE_TOKEN_TO_CATEGORY)
_TOKEN_RE = re.compile(
    r"(\d+)" + re.escape(ANCHOR_SEP_TOKEN) + r"(.*?)(" + _TYPE_TOKEN_ALT + r")",
    re.DOTALL,
)


def parse_spans(text):
    """Parse (anchor_id, category, span_text) triples from model or gold
    output. Returns [] for a genuinely empty response -- the training target
    for a chunk with no gold spans really is the empty string, so an empty
    generation is a valid zero-entity prediction, not a parse failure.
    Returns None only when there's non-empty text that still fails to match
    the expected format at all (truncated mid-span, degenerate output, etc).
    Stateless: every match is a complete record on its own."""
    text = text.strip().removesuffix("<turn|>").strip()
    if not text:
        return []
    results = [
        (int(anchor_digits), _TYPE_TOKEN_TO_CATEGORY[type_token], span_text)
        for anchor_digits, span_text, type_token in _TOKEN_RE.findall(text)
    ]
    return results or None


def pooled_prf(tp, n_pred, n_gold):
    prec = tp / n_pred if n_pred else 0.0
    rec = tp / n_gold if n_gold else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


# %% [markdown] id="22a"
# ### Repetition loops
#
# Generation can loop -- repeating the same `(category, text)` pair, or
# cycling through a short sequence of a few pairs, until the token budget
# runs out. `GEN_STREAK_LIMIT`/`GEN_MAX_PERIOD` (Control variables cell)
# detect this via `_max_periodic_repeat` below; the printed eval stats
# report how often it happens (`over_streak_limit`).

# %% id="22b"
def _max_periodic_repeat(items, max_period):
    """Longest run where a period-k block repeats consecutively -- catches a
    cycling loop (A,A,B,A,A,B,...), not just the same item back-to-back."""
    n = len(items)
    best_period, best_cycles, best_start = 0, 0, -1
    for k in range(1, max_period + 1):
        i = 0
        while i + 2 * k <= n:
            if items[i:i + k] == items[i + k:i + 2 * k]:
                cycles = 2
                j = i + 2 * k
                while j + k <= n and items[j:j + k] == items[i:i + k]:
                    cycles += 1
                    j += k
                covered = k * cycles
                if covered > best_period * best_cycles or (
                    covered == best_period * best_cycles and k < best_period
                ):
                    best_period, best_cycles, best_start = k, cycles, i
                i = j
            else:
                i += 1
    return best_period, best_cycles, best_start


# Deliberately its own name, not a rebind of CATEGORIES: CATEGORIES (set
# above from the dataset's declared order) is baked into SYSTEM_PROMPT and
# TYPE_TOKEN_BY_CATEGORY, so reassigning it here -- even to the same set in a
# different order -- would silently desync the prompt/token map from every
# table below if this cell were ever re-executed after them.
TABLE_CATEGORIES = sorted(set(_TYPE_TOKEN_TO_CATEGORY.values()))

# %% [markdown] id="22c"
# ### Loop restart
#
# `GEN_RESTART_ON_LOOP` (Control variables cell, on by default): when
# a loop is detected, or the first pass is fully unparseable, generation is
# cut at the last trustworthy anchor and re-issued as a fresh, independent
# generation over the remaining text, up to `GEN_MAX_RESTARTS` times.

# %% id="23"
def _insert_anchor_markers(text, anchors, anchor_ids):
    out = text
    for idx in range(len(anchors), 0, -1):
        pos = anchors[idx - 1]
        out = out[:pos] + f"{ANCHOR_TOKEN}{anchor_ids[idx - 1]}{ANCHOR_CLOSE_TOKEN}" + out[pos:]
    return out


def _loop_cut_index(pairs, max_period=None, streak_limit=None):
    seq = [(t, s) for _, t, s in pairs]
    period, cycles, start = _max_periodic_repeat(seq, GEN_MAX_PERIOD if max_period is None else max_period)
    limit = GEN_STREAK_LIMIT if streak_limit is None else streak_limit
    return (start, period, cycles) if period * cycles > limit else None


def _generate_once(m, system_content, user_content, generation_config):
    prompt = processor.apply_chat_template(
        [{"role": "system", "content": system_content},
         {"role": "user", "content": user_content}],
        tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(m.device)
    _rng = torch.random.get_rng_state()
    _cuda_rng = torch.cuda.get_rng_state_all() if HAS_GPU else None
    try:
        with torch.no_grad(), torch.autocast(device_type=m.device.type, dtype=TORCH_DTYPE):
            out = m.generate(**inputs, generation_config=generation_config)
    finally:
        torch.random.set_rng_state(_rng)
        if HAS_GPU:
            torch.cuda.set_rng_state_all(_cuda_rng)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

# %% [markdown] id="23b"
# `generate_with_restart` is the driver: detects a loop via `_loop_cut_index`,
# then re-issues generation over the untrusted remainder via `_generate_once`
# on a fresh, renumbered mini-chunk (`_insert_anchor_markers`).

# %% id="23c"
def generate_with_restart(m, first_pairs, chunk_text, anchors, anchor_ids, system_content,
                          generation_config, max_restarts=None, first_hit_max_tokens=False):
    """On a detected loop (or a fully-unparseable first pass), cuts at the
    last trustworthy anchor and re-generates the remainder as a fresh,
    self-contained mini-chunk. Returns (pairs, info)."""
    max_restarts = GEN_MAX_RESTARTS if max_restarts is None else max_restarts
    info = {"restarts": 0, "triggered": False, "still_looping": False, "max_tokens_triggered": False}
    id_to_index = anchor_id_to_index(anchor_ids)
    pairs, last_cut = first_pairs, 0

    for restart_i in range(max_restarts):
        if pairs is None:
            next_index, cut_index = 1, 0
        else:
            hit = _loop_cut_index(pairs)
            if hit is None:
                if restart_i == 0 and first_hit_max_tokens and GEN_RESTART_ON_MAX_TOKENS:
                    next_index, cut_index = 1, 0
                    info["max_tokens_triggered"] = True
                else:
                    break
            else:
                cut_index, _, _ = hit
                # .get(..., 0) + 1 -> falls back to restarting from the very
                # beginning (same as the cut_index==0 case) if the model
                # hallucinated an anchor id that was never actually assigned
                # to this chunk -- a real possibility, since this id comes
                # straight from the model's own (possibly degenerate) text.
                next_index = (1 if cut_index == 0
                              else id_to_index.get(pairs[cut_index - 1][0], 0) + 1)
        info["triggered"] = True

        next_index = max(next_index, last_cut + 1)
        if next_index > len(anchors):
            break

        offset = anchors[next_index - 1]
        sub_anchors = [p - offset for p in anchors[next_index - 1:]]
        sub_anchor_ids = assign_anchor_ids(sub_anchors, _ANCHOR_RNG)
        sub_text = _insert_anchor_markers(chunk_text[offset:], sub_anchors, sub_anchor_ids)
        sub_gen = _generate_once(m, system_content, sub_text, generation_config)

        info["restarts"] += 1
        last_cut = next_index

        prefix = [] if pairs is None else [
            p for i, p in enumerate(pairs)
            if i < cut_index and id_to_index.get(p[0], float("inf")) < next_index]
        sub_pairs = parse_spans(sub_gen)
        if sub_pairs is None:
            pairs = prefix or None
            break
        sub_pos_to_orig_id = {sub_anchor_ids[i]: anchor_ids[next_index - 1 + i]
                               for i in range(len(sub_anchors))}
        pairs = prefix + [(sub_pos_to_orig_id.get(a, a), t, s) for a, t, s in sub_pairs]
    else:
        info["still_looping"] = pairs is not None and _loop_cut_index(pairs) is not None

    return pairs, info

# %% [markdown] id="23a"
# ### Span reconstruction
#
# The model's raw output only carries an anchor id, a category, and the
# copied span text -- no character offsets. To score true positional
# (char-level) overlap, each predicted span's text is relocated onto the
# original, un-anchored chunk text via a DP alignment (`construct_spans` /
# `reconstruct_pred_spans` below), constrained to its own anchor's region
# (case-insensitive, whole-token match). A span that can't be placed there
# counts as unplaced (scored as a false positive, by its length).

# %% id="24"
def _boundary_ok(low, q, L, N, needle):
    left_ok = q == 0 or not (low[q - 1].isalnum() and needle[0].isalnum())
    right_ok = q + L == N or not (low[q + L].isalnum() and needle[-1].isalnum())
    return left_ok and right_ok


def construct_spans(note, items):
    low = note.lower()
    m = len(items)
    N = len(note)
    lens = [len(s) for s in items]
    lows = [s.lower() for s in items]

    def earliest_match(j, p):
        L = lens[j]
        if L == 0:
            return None
        q = low.find(lows[j], p)
        while q != -1:
            if _boundary_ok(low, q, L, N, lows[j]):
                return q
            q = low.find(lows[j], q + 1)
        return None

    f = [[0] * (m + 1) for _ in range(N + 2)]
    for j in range(m - 1, -1, -1):
        L = lens[j]
        for p in range(N, -1, -1):
            best = f[p][j + 1]
            q0 = earliest_match(j, p)
            if q0 is not None:
                cand = 1 + f[q0 + L][j + 1]
                if cand > best:
                    best = cand
            f[p][j] = best

    spans = []
    pos = 0
    for j in range(m):
        L = lens[j]
        q0 = earliest_match(j, pos)
        if q0 is not None and 1 + f[q0 + L][j + 1] == f[pos][j]:
            spans.append((q0, q0 + L))
            pos = q0 + L
        else:
            spans.append(None)
    return spans

# %% [markdown] id="24a"
# ### Scoring helpers
#
# Relocate each predicted span onto its anchor's own region
# (`reconstruct_pred_spans`, using `construct_spans` above), then reduce to
# char-set precision/recall/F1 (`char_counts`).

# %% id="24b"
def reconstruct_pred_spans(pred_triples, chunk_text, anchors, id_to_index):
    n = len(anchors)
    groups, order = {}, []
    for aid, cat, text in pred_triples:
        if aid not in groups:
            groups[aid] = []
            order.append(aid)
        groups[aid].append((cat, text))

    placed, unplaced = [], []
    for aid in order:
        items = groups[aid]
        idx = id_to_index.get(aid)
        if idx is None or idx < 1 or idx > n:
            unplaced.extend(items)
            continue
        region_start = anchors[idx - 1]
        region_end = anchors[idx] if idx < n else len(chunk_text)
        region = chunk_text[region_start:region_end]
        spans = construct_spans(region, [t for _, t in items])
        for (cat, text), sp in zip(items, spans):
            if sp is None:
                unplaced.append((cat, text))
            else:
                placed.append((region_start + sp[0], region_start + sp[1], cat))
    return placed, unplaced


def _char_indices(spans):
    idx = set()
    for sp in spans:
        idx.update(range(sp[0], sp[1]))
    return idx


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def char_counts(placed, unplaced, gold):
    P, G = _char_indices(placed), _char_indices(gold)
    pen = sum(len(t) for _, t in unplaced)
    tp = len(P & G)
    fp = len(P - G) + pen
    fn = len(G - P)
    return tp, fp, fn

# %% [markdown] id="24c"
# ### Metrics accumulation
#
# Per-category + overall running totals (multiset TP/pred/gold for the
# category-match and word-overlap scales, char TP/FP/FN for the positional
# scale), reduced to micro/macro P/R/F1 by `_finalize_scale`.

# %% id="25"
def _new_metrics_row(positional=False):
    row = {"tp": 0, "n": 0}
    if positional:
        row["fp"] = 0
        row["fn"] = 0
    else:
        row["n_pred"] = 0
        row["n_gold"] = 0
    return row


def _accum_multiset(row, pred, gold):
    p, g = Counter(pred), Counter(gold)
    tp = sum((p & g).values())
    n_pred, n_gold = sum(p.values()), sum(g.values())
    row["tp"] += tp; row["n_pred"] += n_pred; row["n_gold"] += n_gold
    row["n"] += 1


def _accum_positional(row, placed, unplaced, gold):
    tp, fp, fn = char_counts(placed, unplaced, gold)
    row["tp"] += tp; row["fp"] += fp; row["fn"] += fn
    row["n"] += 1


def _finalize_row(row):
    if "fp" in row:
        micro = _prf(row["tp"], row["fp"], row["fn"])
    else:
        micro = pooled_prf(row["tp"], row["n_pred"], row["n_gold"])
    return {
        "micro": dict(zip(("precision", "recall", "f1"), micro)),
        "n": row["n"],
    }


def _finalize_scale(scale_pools, is_positional):
    """Micro is always the per-category pools summed, never a separately
    accumulated category-agnostic pool -- for the positional scale
    specifically, that distinction matters: a wrong-category prediction
    landing on the right characters must not count as a true positive."""
    cats = {cat: _finalize_row(scale_pools[cat]) for cat in TABLE_CATEGORIES}
    tp = sum(scale_pools[c]["tp"] for c in TABLE_CATEGORIES)
    rows = {}
    if is_positional:
        fp = sum(scale_pools[c]["fp"] for c in TABLE_CATEGORIES)
        fn = sum(scale_pools[c]["fn"] for c in TABLE_CATEGORIES)
        p, r, f = _prf(tp, fp, fn)
        rows["micro"] = {"precision": p, "recall": r, "f1": f}
    else:
        n_pred = sum(scale_pools[c]["n_pred"] for c in TABLE_CATEGORIES)
        n_gold = sum(scale_pools[c]["n_gold"] for c in TABLE_CATEGORIES)
        p, r, f = pooled_prf(tp, n_pred, n_gold)
        rows["micro"] = {"precision": p, "recall": r, "f1": f}
    for cat in TABLE_CATEGORIES:
        rows[cat] = {**cats[cat]["micro"], "n": cats[cat]["n"]}
    rows["macro"] = {
        col: sum(rows[cat][col] for cat in TABLE_CATEGORIES) / len(TABLE_CATEGORIES)
        for col in ("precision", "recall", "f1")
    }
    return rows


def _metrics_table_rows(rows, columns=("precision", "recall", "f1")):
    order = [("micro", 0), ("macro", 0)] + [(c, 1) for c in TABLE_CATEGORIES]
    for key, indent in order:
        row = rows.get(key, {})
        yield key, indent, [row.get(col) for col in columns]

# %% [markdown] id="25a"
# ### Table rendering

# %% id="25b"
from tabulate import tabulate

_EVAL_TABLE_HDR = {"precision": "P", "recall": "R", "f1": "F1"}


def _render_metrics_table(rows, columns=("precision", "recall", "f1")):
    headers = ["category"] + [_EVAL_TABLE_HDR.get(c, c) for c in columns]
    table_rows = []
    for key, indent, values in _metrics_table_rows(rows, columns):
        name = ("  " if indent else "") + key
        cells = [f"{v:.3f}" if v is not None else "--" for v in values]
        table_rows.append([name] + cells)
    return tabulate(table_rows, headers=headers, tablefmt="simple",
                     colalign=("left",), disable_numparse=True)


def _print_eval_tables(label, n, valid, invalid, invalid_rate, tables,
                        hit_max_tokens=0, hit_max_tokens_rate=0.0,
                        over_streak_limit=0, over_streak_limit_rate=0.0,
                        restart_enabled=False, restarted=0, restarted_rate=0.0,
                        restart_still_looping=0):
    header = f"{label.upper()} -- Chunk evaluation" if label else "Chunk evaluation"
    print(header)
    print(f"{n} examples | {valid} valid | {invalid} invalid ({100*invalid_rate:.1f}%)")
    print(f"  hit max_new_tokens: {hit_max_tokens} ({100*hit_max_tokens_rate:.1f}%) | "
          f"exceeded streak limit ({GEN_STREAK_LIMIT}): {over_streak_limit} "
          f"({100*over_streak_limit_rate:.1f}%)")
    if restart_enabled:
        print(f"  loop restarts (GEN_RESTART_ON_LOOP): {restarted} example(s) "
              f"({100*restarted_rate:.1f}%) | still looping after restart budget: "
              f"{restart_still_looping}")
    for title, rows, columns in tables:
        print(f"\n{title}")
        print(_render_metrics_table(rows, columns))

# %% id="26"
def evaluate_model(m, dataset, n_eval, label="", reconstruct=True, second_table="word",
                   generation_config=None, allow_restart=True, print_tables=True):
    """Generate over `dataset[:n_eval]` (the whole dataset if n_eval is None)
    and score. Returns a metrics dict with res["scale1_rows"] (category+text,
    the headline number), res["word_rows"], and res["positional_rows"]
    (char-level, if reconstruct) -- plus res["scale1_counts"] and
    res["positional_counts"], the raw tp/fp/fn behind those two, exposed so
    a caller can charge structural misses and recompute without re-running.

    `print_tables=False` still prints the generation-health stats (invalid/
    restart rate) but skips the P/R/F1 tables themselves -- for a caller that
    charges structural misses afterward (the Reproduce cell below) and wants
    the printed output to show only the charged tables, not the unadjusted
    ones too."""
    was_training = m.training
    prev_cache = m.config.use_cache
    m.config.use_cache = True
    m.eval()
    _gen_cfg = generation_config if generation_config is not None else gen_cfg
    torch.manual_seed(SEED)

    n = len(dataset) if n_eval is None else min(n_eval, len(dataset))
    invalid = 0
    hit_max_tokens = 0
    over_streak_limit = 0
    restart_active = GEN_RESTART_ON_LOOP and allow_restart and GEN_MAX_RESTARTS > 0
    restarted = 0
    restart_still_looping = 0

    pools = {
        "multiset":   {row: _new_metrics_row() for row in TABLE_CATEGORIES},
        "word":       {row: _new_metrics_row() for row in TABLE_CATEGORIES},
        "positional": {row: _new_metrics_row(positional=True) for row in TABLE_CATEGORIES}
                      if reconstruct else None,
    }

    prev_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for batch_start in tqdm(range(0, n, EVAL_GEN_BATCH_SIZE), desc="Eval"):
            batch_idx = range(batch_start, min(batch_start + EVAL_GEN_BATCH_SIZE, n))
            batch_msgs = [dataset[i]["messages"] for i in batch_idx]
            batch_gold = [parse_spans(msgs[2]["content"]) or [] for msgs in batch_msgs]
            batch_recon = [(dataset[i]["chunk_text"], list(dataset[i]["anchors"]),
                            list(dataset[i]["anchor_ids"]),
                            json.loads(dataset[i]["gold_spans"])) for i in batch_idx]
            batch_restart = [(dataset[i]["chunk_text"], list(dataset[i]["anchors"]),
                               list(dataset[i]["anchor_ids"]))
                              for i in batch_idx] if restart_active else [None] * len(batch_msgs)

            prompts, batch_sys = [], []
            for msgs in batch_msgs:
                prompt_msgs = msgs[:2]
                batch_sys.append(prompt_msgs[0]["content"])
                prompts.append(processor.apply_chat_template(
                    prompt_msgs, tokenize=False, add_generation_prompt=True
                ))

            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(m.device)
            with torch.no_grad(), torch.autocast(device_type=m.device.type, dtype=TORCH_DTYPE):
                out = m.generate(**inputs, generation_config=_gen_cfg)
            prompt_len = inputs["input_ids"].shape[1]

            for row, gold_pairs, recon, sys_content, rst in zip(
                    out, batch_gold, batch_recon, batch_sys, batch_restart):
                gen_text = tokenizer.decode(row[prompt_len:], skip_special_tokens=True)
                _row_hit_max_tokens = not any(t.item() in _gen_cfg.eos_token_id for t in row[prompt_len:])
                if _row_hit_max_tokens:
                    hit_max_tokens += 1

                pred_pairs = parse_spans(gen_text)

                if restart_active:
                    _chunk_text_r, _anchors_r, _anchor_ids_r = rst
                    pred_pairs, _rinfo = generate_with_restart(
                        m, pred_pairs, _chunk_text_r, _anchors_r, _anchor_ids_r,
                        sys_content, _gen_cfg, first_hit_max_tokens=_row_hit_max_tokens)
                    if _rinfo["restarts"]:
                        restarted += 1
                    if _rinfo["still_looping"]:
                        restart_still_looping += 1

                if pred_pairs is None:
                    invalid += 1
                else:
                    pred_type = [(t, s) for _, t, s in pred_pairs]
                    _period, _cycles, _ = _max_periodic_repeat(pred_type, GEN_MAX_PERIOD)
                    if _period * _cycles > GEN_STREAK_LIMIT:
                        over_streak_limit += 1

                # An invalid (unparseable) response is scored as an empty
                # prediction against the real gold spans, on every scale below
                # -- a full miss, not skipped. Skipping it would drop those
                # gold spans from the denominator entirely instead of costing
                # recall, silently inflating every table on invalid-heavy runs.
                pred_pairs_scored = pred_pairs if pred_pairs is not None else []

                if reconstruct:
                    chunk_text, anchors, anchor_ids, gold_spans = recon
                    if pred_pairs is not None:
                        placed, unplaced = reconstruct_pred_spans(
                            pred_pairs, chunk_text, anchors, anchor_id_to_index(anchor_ids))
                    else:
                        placed, unplaced = [], []

                for cat in TABLE_CATEGORIES:
                    gold_cat_pairs = [s for _, t, s in gold_pairs if t == cat]
                    pred_cat_pairs = [s for _, t, s in pred_pairs_scored if t == cat]
                    if gold_cat_pairs or pred_cat_pairs:
                        _accum_multiset(pools["multiset"][cat], pred_cat_pairs, gold_cat_pairs)
                        gold_cat_words = [w for s in gold_cat_pairs for w in re.findall(r"\w+", s.lower())]
                        pred_cat_words = [w for s in pred_cat_pairs for w in re.findall(r"\w+", s.lower())]
                        _accum_multiset(pools["word"][cat], pred_cat_words, gold_cat_words)
                    if reconstruct:
                        pl = [(s, e) for s, e, c in placed if c == cat]
                        gl = [(s, e) for s, e, c in gold_spans if c == cat]
                        up = [(c2, t) for c2, t in unplaced if c2 == cat]
                        if pl or gl or up:
                            _accum_positional(pools["positional"][cat], pl, up, gl)
        valid = n - invalid
        res = {
            "n_eval": n, "invalid": invalid, "invalid_rate": invalid / n if n else 0.0,
            "hit_max_tokens": hit_max_tokens, "hit_max_tokens_rate": hit_max_tokens / n if n else 0.0,
            "over_streak_limit": over_streak_limit,
            "over_streak_limit_rate": over_streak_limit / n if n else 0.0,
            "restart_enabled": restart_active, "restarted": restarted,
            "restarted_rate": restarted / n if n else 0.0,
            "restart_still_looping": restart_still_looping,
        }
        res["scale1_rows"] = _finalize_scale(pools["multiset"], is_positional=False)
        # Raw pooled counts behind scale1_rows -- exposed so a caller can charge
        # structural misses (spans the method never had a chance to predict, see
        # the chunking and test-split length-check cells above) as zero-credit
        # FN and recompute, without re-running eval.
        res["scale1_counts"] = {
            cat: {"tp": pools["multiset"][cat]["tp"], "n_pred": pools["multiset"][cat]["n_pred"],
                  "n_gold": pools["multiset"][cat]["n_gold"]}
            for cat in TABLE_CATEGORIES
        }
        res["word_rows"] = _finalize_scale(pools["word"], is_positional=False)
        if reconstruct:
            res["positional_rows"] = _finalize_scale(pools["positional"], is_positional=True)
            # Raw tp/fp/fn behind positional_rows -- same purpose as scale1_counts
            # above, for charging structural misses on the char-level table.
            res["positional_counts"] = {
                cat: {"tp": pools["positional"][cat]["tp"], "fp": pools["positional"][cat]["fp"],
                      "fn": pools["positional"][cat]["fn"]}
                for cat in TABLE_CATEGORIES
            }

        # Always print the header/stats -- even at 0 valid examples, seeing the
        # invalid-rate/streak/restart stats is more useful for diagnosing a
        # broken run than nothing. The P/R/F1 tables themselves are gated on
        # print_tables (see docstring).
        tables = []
        if print_tables:
            tables.append(("Category match", res["scale1_rows"], ("precision", "recall", "f1")))
            if second_table == "word":
                tables.append(("Word overlap", res["word_rows"], ("precision", "recall", "f1")))
            elif second_table == "positional":
                tables.append(("Positional", res["positional_rows"], ("precision", "recall", "f1")))
        _print_eval_tables(label, n, valid, invalid, res["invalid_rate"], tables,
                           hit_max_tokens, res["hit_max_tokens_rate"],
                           over_streak_limit, res["over_streak_limit_rate"],
                           restart_active, restarted, res["restarted_rate"],
                           restart_still_looping)
        return res
    finally:
        # Covers the whole function body, not just the generation loop --
        # model/tokenizer state must be restored even if an exception comes
        # from metrics finalization or table rendering, not just generation.
        tokenizer.padding_side = prev_padding_side
        m.config.use_cache = prev_cache
        if was_training:
            m.train()

# %% [markdown] id="25c"
# ### Structural-miss charging
#
# Defined here, ahead of the sanity check below, so a sanity-generation
# failure (e.g. an OOM) can never leave these undefined for the Reproduce
# cell further down to fail on with an unrelated `NameError`.

# %% id="25d"
def charge_structural_misses(scale1_counts, dropped_by_category):
    """Category-match table (see evaluate_model's scale1_counts) recomputed
    with structurally-missed gold spans (boundary-straddling drops,
    truncated-away spans -- see the chunking and test-split length-check
    cells above) counted as zero-credit misses instead of excluded.
    Comparable to literature numbers computed over the full official test
    set, not just the subset this chunking scheme could represent."""
    rows = {}
    for cat in TABLE_CATEGORIES:
        c = scale1_counts[cat]
        n_gold = c["n_gold"] + dropped_by_category.get(cat, 0)
        p, r, f = pooled_prf(c["tp"], c["n_pred"], n_gold)
        rows[cat] = {"precision": p, "recall": r, "f1": f}
    tp = sum(scale1_counts[c]["tp"] for c in TABLE_CATEGORIES)
    n_pred = sum(scale1_counts[c]["n_pred"] for c in TABLE_CATEGORIES)
    n_gold = sum(scale1_counts[c]["n_gold"] + dropped_by_category.get(c, 0) for c in TABLE_CATEGORIES)
    p, r, f = pooled_prf(tp, n_pred, n_gold)
    rows["micro"] = {"precision": p, "recall": r, "f1": f}
    rows["macro"] = {
        col: sum(rows[cat][col] for cat in TABLE_CATEGORIES) / len(TABLE_CATEGORIES)
        for col in ("precision", "recall", "f1")
    }
    return rows


def charge_structural_misses_positional(positional_counts, dropped_chars_by_category):
    """Positional (char-level) table (see evaluate_model's positional_counts)
    recomputed with structurally-missed gold spans' total char length (see
    the chunking and test-split length-check cells above) added to each
    category's FN instead of excluded -- the char-level counterpart to
    charge_structural_misses above (a dropped span costs recall in
    proportion to its length, not just as one more miss)."""
    rows = {}
    for cat in TABLE_CATEGORIES:
        c = positional_counts[cat]
        fn = c["fn"] + dropped_chars_by_category.get(cat, 0)
        p, r, f = _prf(c["tp"], c["fp"], fn)
        rows[cat] = {"precision": p, "recall": r, "f1": f}
    tp = sum(positional_counts[c]["tp"] for c in TABLE_CATEGORIES)
    fp = sum(positional_counts[c]["fp"] for c in TABLE_CATEGORIES)
    fn = sum(positional_counts[c]["fn"] + dropped_chars_by_category.get(c, 0) for c in TABLE_CATEGORIES)
    p, r, f = _prf(tp, fp, fn)
    rows["micro"] = {"precision": p, "recall": r, "f1": f}
    rows["macro"] = {
        col: sum(rows[cat][col] for cat in TABLE_CATEGORIES) / len(TABLE_CATEGORIES)
        for col in ("precision", "recall", "f1")
    }
    return rows

# %% [markdown] id="26a"
# ## Sanity check
#
# A full run can take well over an hour on a T4. If something's actually
# wrong (wrong prompt, format mismatch, a stale/incompatible adapter), you
# want to know from ONE example, not after waiting for all of `N_EVAL`. Runs
# a single generation and prints it raw, next to the gold target -- if this
# doesn't look like `id<unused2>text<TYPE_TOKEN>id<unused2>text<TYPE_TOKEN>...`,
# something is genuinely broken (wrong prompt variant for this adapter, a
# tokenizer/model version mismatch, or this adapter using a different output
# format than this script expects) -- don't run the full eval until this
# looks right.

# %% id="26b"
_sanity_row = test_ds[0]
_sanity_gen = _generate_once(
    model, _sanity_row["messages"][0]["content"], _sanity_row["messages"][1]["content"], gen_cfg
)
print("RAW MODEL OUTPUT:")
print(repr(_sanity_gen[:500]))
print("\nGOLD TARGET:")
print(repr(_sanity_row["messages"][2]["content"][:500]))
print("\nParsed:", parse_spans(_sanity_gen))

# %% [markdown] id="27"
# ## Reproduce test-set results
#
# Evaluates the real, held-out test split ONCE, at the fixed operating point
# set in the Control variables cell (greedy decoding by default -- every
# model card's headline number is reported at a single fixed temperature,
# not a sweep).
#
# Two tables are printed, both charged for structural misses -- gold spans
# that never had a chance to be predicted at all, dropped for straddling a
# chunk boundary or lost to truncating an over-length chunk (during the
# test-set length check above), recomputed as zero-credit misses instead of
# excluded: category-match
# (`TEST_STRUCTURAL_FN_BY_CATEGORY`, charged by span count) and positional
# (`TEST_STRUCTURAL_FN_CHARS_BY_CATEGORY`, charged by total char length, so
# a dropped span costs recall in proportion to its size). Both are
# comparable to literature results computed over the full official test
# set. The unadjusted word-overlap table is still computed and saved to
# `eval_results_<dataset>_<timestamp>.json` below, just not printed here.

# %% id="28"
with gpu_mem_block("eval"):
    results = evaluate_model(
        model, test_ds, N_EVAL, label="test",
        reconstruct=True, second_table="positional",
        generation_config=gen_cfg, allow_restart=True,
        print_tables=False,
    )

results["scale1_rows_charged"] = charge_structural_misses(
    results["scale1_counts"], TEST_STRUCTURAL_FN_BY_CATEGORY
)
results["positional_rows_charged"] = charge_structural_misses_positional(
    results["positional_counts"], TEST_STRUCTURAL_FN_CHARS_BY_CATEGORY
)
print("Category match (charged for structural misses)")
print(_render_metrics_table(results["scale1_rows_charged"]))
print("\nPositional (charged for structural misses)")
print(_render_metrics_table(results["positional_rows_charged"]))

# %% id="29"
from datetime import datetime
from importlib.metadata import version as _pkg_version, PackageNotFoundError

_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")   # local time -- matches Colab's default clock

def _pkg_versions(names):
    # PackageNotFoundError -> None rather than raising: a genuinely uninstalled
    # package would already have failed far earlier (import time), so this is
    # only ever a distribution-name mismatch (e.g. import name vs. PyPI name).
    out = {}
    for name in names:
        try:
            out[name] = _pkg_version(name)
        except PackageNotFoundError:
            out[name] = None
    return out

# Provenance -- everything needed to tell two result files apart without
# guessing from the filename or a separately-kept console log: which
# hardware/dtype/software actually produced these numbers (relevant given
# the fp16-vs-bf16 run-to-run variance noted elsewhere), every generation
# knob that can change output (not just the ones already exercised by
# default), exactly which pinned/resolved artifacts were used, and the
# structural-miss counts that would otherwise only ever appear as a printed
# line, never saved.
results["metadata"] = {
    "timestamp": _timestamp,
    "dataset": DATASET,
    "adapter_id": EVAL_ADAPTER_ID,
    "hub_branch": HUB_BRANCH,
    "adapter_commit": _ADAPTER_COMMIT,   # resolved commit HUB_BRANCH pointed at for this run
    "model_size": MODEL_SIZE,
    "base_model_revision": BASE_MODEL_REVISION,   # None means it fell back to unpinned -- see the "Resolve base model" cell
    "corpus_sha256": _CORPUS_SHA256[DATASET],
    "attn_implementation": ATTN_IMPLEMENTATION,
    "gpu_name": torch.cuda.get_device_name(0),
    "torch_dtype": str(TORCH_DTYPE),
    "package_versions": _pkg_versions(
        ["transformers", "torch", "peft", "accelerate", "bitsandbytes", "datasets"]
    ),
    "seed": SEED,
    "max_length": MAX_LENGTH,
    "eval_gen_batch_size": EVAL_GEN_BATCH_SIZE,
    "anchor_max_section_chars": ANCHOR_MAX_SECTION_CHARS,
    "max_new_tokens": gen_cfg.max_new_tokens,
    "gen_do_sample": GEN_DO_SAMPLE,
    "gen_temperature": GEN_TEMPERATURE if GEN_DO_SAMPLE else None,
    "gen_top_p": GEN_TOP_P,
    "gen_top_k": GEN_TOP_K,
    "gen_no_repeat_ngram_size": GEN_NO_REPEAT_NGRAM_SIZE,
    "gen_streak_limit": GEN_STREAK_LIMIT,
    "gen_max_period": GEN_MAX_PERIOD,
    "gen_restart_on_loop": GEN_RESTART_ON_LOOP,
    "gen_max_restarts": GEN_MAX_RESTARTS,
    "gen_restart_on_max_tokens": GEN_RESTART_ON_MAX_TOKENS,
    "n_truncated": _n_truncated,
    "n_dropped_after_truncation_retry": _n_dropped_after_truncation_retry,
    "test_structural_fn_by_category": dict(TEST_STRUCTURAL_FN_BY_CATEGORY),
    "test_structural_fn_chars_by_category": dict(TEST_STRUCTURAL_FN_CHARS_BY_CATEGORY),
}

_out_path = Path(f"eval_results_{DATASET}_{_timestamp}.json")
_out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nWrote {_out_path}")
