# LINDA-LLM Extraction Toolkit

Production-ready scripts for extracting molecular interaction triples — protein–protein interactions (PPI) and transcription factor–target gene (TF) regulations — from biomedical text using large language models.

This directory is the `llm_extractions/` component of the public [dieterich-lab/LLM_Relations](https://github.com/dieterich-lab/LLM_Relations) repository.

---

## Contents

- [Overview](#overview)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running Extractions](#running-extractions)
  - [Basic Usage](#basic-usage)
  - [Full CLI Reference](#full-cli-reference)
  - [Extraction Modes](#extraction-modes)
  - [Prompt Strategies](#prompt-strategies)
  - [In-Context Examples](#in-context-examples)
  - [Dynamic Example Retrieval (DynEx)](#dynamic-example-retrieval-dynex)
  - [STRING Knowledge Lookup](#string-knowledge-lookup)
  - [Ensemble Voting](#ensemble-voting)
  - [Tree of Thoughts (ToT)](#tree-of-thoughts-tot)
- [Output Format and Directory Layout](#output-format-and-directory-layout)
- [SLURM / HPC Usage](#slurm--hpc-usage)
- [Synonym Generation](#synonym-generation)
- [Fine-Tuning](#fine-tuning)
- [Datasets](#datasets)
- [Model Hosting](#model-hosting)

---

## Overview

The toolkit wraps any Ollama-hosted LLM (or OpenAI-compatible endpoint) with a configurable biomedical relation-extraction pipeline. A single entry-point script (`extract.py`) controls the full workflow:

1. **Load documents** from one of several biomedical corpora.
2. **Build a prompt** that combines a biomedical system instruction, optional in-context examples (static or dynamically retrieved), and optionally background knowledge from the STRING database.
3. **Call an LLM** through a BAML-typed interface to obtain structured JSON output.
4. **Aggregate results** via optional ensemble voting or Tree-of-Thoughts (ToT) multi-path reasoning.
5. **Save** every extraction as a `.jsonl` file, one record per document, in a fully reproducible hierarchical directory.

The same framework supports three extraction targets out of the box:

| Target | Description |
|--------|-------------|
| `ppi`  | Protein–protein interactions (direct physical interactions) |
| `tf`   | Transcription factor → target gene regulatory relations |
| `ppitf`| Combined PPI + TF extraction |

---

## Repository Layout

```
scripts/
├── extract.py            # Main CLI entry point
├── parser.py             # Centralised argument definitions
├── paths.py              # Reproducible, hierarchical output paths
├── clients.py            # LLM client registry (model aliases → endpoints)
├── prompts.py            # Prompt templates and in-context examples
├── extraction_utils.py   # NER, extraction, ensemble & ToT logic
├── rag_utils.py          # Dynamic example retrieval (DynEx) via HNSW index
├── embed.py              # Build dense vector indices for DynEx
├── synonyms.py           # Generate protein synonym dictionaries
├── ppi_lookup.py         # STRING database fuzzy lookup
├── dataset.py            # Dataset loading and finetuning-format conversion
├── documents.py          # Document loading, caching, and chunking
├── brat_utils.py         # BRAT annotation format parser
├── finetune.py           # LoRA supervised finetuning
├── finetuning_tools.py   # Utilities for finetuning
├── baml/
│   ├── baml_src/         # BAML schema files (rel.baml, names.baml, …)
│   └── baml_client/      # Auto-generated typed Python client
├── slurm/                # Ready-to-submit SLURM batch scripts
│   ├── regu_ppi_matrix.sh          # Full regulatome PPI experiment matrix
│   ├── regu_ppi_synonyms.sh        # Synonym generation run
│   └── …
├── pyproject.toml        # Poetry dependency manifest
└── poetry.lock
```

---

## Installation

Requires **Python 3.11** and [Poetry](https://python-poetry.org/) 1.7+.

```bash
# Install Poetry if not present
command -v poetry >/dev/null || pipx install poetry

git clone https://github.com/dieterich-lab/LLM_Relations.git
cd LLM_Relations/llm_extractions

# Install dependencies
poetry install

# Regenerate the BAML typed client (only needed after editing baml_src/)
poetry run baml-cli generate
```

---

## Configuration

All runtime paths can be overridden via environment variables. Defaults are relative to the repository root and require no configuration for a standard clone.

| Variable | Default | Purpose |
|----------|---------|---------|
| `LINDA_LLM_PROJECT_ROOT` | repository root | Base for all relative defaults |
| `LINDA_LLM_OUTPUT_ROOT` | `{PROJECT_ROOT}/outputs/` | Root for all generated artefacts |
| `LINDA_LLM_TRIPLES_ROOT` | `{OUTPUT_ROOT}/triples/` | Base folder for extracted triples |
| `LINDA_LLM_REGULATOME_ROOT` | `{PROJECT_ROOT}/RegulaTome/` | RegulaTome corpus and annotations |
| `LINDA_LLM_RESOURCES_ROOT` | `{PROJECT_ROOT}/resources/` | Shared resources (UniProt tables, etc.) |

Copy `.env.example` to `.env` and fill in your paths — `paths.py` loads it automatically via `python-dotenv`:

```bash
cp .env.example .env
# edit .env with your actual paths
```

Alternatively, export variables in your shell before running:

```bash
export LINDA_LLM_REGULATOME_ROOT=/data/RegulaTome
```

`paths.py` reads both `.env` and shell environment at import time and creates any missing output directories automatically.

---

## Running Extractions

### Basic Usage

```bash
python extract.py \
  --model  llama33 \
  --data   regulatome \
  --target ppi \
  --extractionmode direct \
  --chattype oneshot \
  --doclevel docs
```

This runs the simplest possible configuration: single-call (oneshot) direct extraction using Llama 3.3 70B on the full RegulaTome PPI corpus, saving results under `outputs/triples/regulatome/ppi/llama33/direct/oneshot/docs/`.

---

### Full CLI Reference

#### Model and infrastructure

| Flag | Choices / Default | Description |
|------|-------------------|-------------|
| `--model` | `llama33`* | Model alias (see [Model Hosting](#model-hosting)) |
| `--node` | `g4`* | Node alias that resolves to an Ollama IP address (`g2`–`g5`, `mk22d`, `local`) |
| `--port` | *(auto)* | Override the standard port inferred from `--node` |
| `--nebius` | flag | Route requests to Nebius cloud instead of local Ollama |
| `--apikey` | `NEBIUS_API_KEY_PRP` | Environment variable holding the API key |

#### Data selection

| Flag | Choices / Default | Description |
|------|-------------------|-------------|
| `--data` | `regulatome`* | Corpus to extract from (`regulatome`, `biored`, `cardio`, `5curated`, …) |
| `--target` | `ppi`* | Relation type to extract (`ppi`, `tf`, `ppitf`) |
| `--doclevel` | `docs`* | Process full documents (`docs`) or sliding chunks (`chunks`) |
| `--chunksize` | `2000` | Character length per chunk when using `--doclevel chunks` |
| `--full_corpus` | flag | Use all corpus documents (not just the test split) |
| `--startfromdoc` | `0` | Skip the first N documents |
| `--untildoc` | *(end)* | Stop after document N |
| `--num-shards` / `--shard-index` | — | Split the corpus across parallel workers |

#### Extraction strategy

| Flag | Choices / Default | Description |
|------|-------------------|-------------|
| `--extractionmode` | `direct`* | `direct` (one-stage) or `nerrel` (NER then relation extraction) |
| `--chattype` | `oneshot`* | `oneshot` (single call) or `stepwise` (multi-turn refinement) |
| `--examples` | *(none)* | Add static in-context examples: `pos`, `neg`, or `negpos` |
| `--dynex_k` | `0` | Retrieve *k* semantically similar training examples dynamically |
| `--lookup` | flag | Prepend STRING database background knowledge (forces `nerrel`) |
| `--recall` | flag | Greedy extraction of all candidate relations for later filtering |
| `--force_cot` | flag | Append a chain-of-thought instruction |
| `--noconfidence` | `True`* | Omit per-triple confidence scores |

#### Ensemble and ToT

| Flag | Choices / Default | Description |
|------|-------------------|-------------|
| `--ensemble` | `0` | Run *n* stochastic samples and keep majority-voted triples |
| `--ensemble_temp` | `0.8` | Sampling temperature for ensemble runs |
| `--tot` | `0` | Run *n* reasoning paths (Tree of Thoughts) and combine results |
| `--tot_strategy` | `vote`* | How to combine ToT paths: `vote`, `best`, or `merge` |

#### Named-entity inputs

| Flag | Description |
|------|-------------|
| `--all_nes_given` | Inject all annotated entities from ground-truth files (forces `nerrel`) |
| `--true_nes_given` | Inject only entities that participate in a gold relation (forces `nerrel`) |
| `--spacy_nes_given` | Inject entities predicted by ScispaCy (forces `nerrel`) |

#### Output control

| Flag | Default | Description |
|------|---------|-------------|
| `--ext` | *(none)* | Suffix appended to the output filename |
| `--force_new` | flag | Overwrite existing output files |
| `--loglevel` | `off` | BAML logging verbosity (`off`, `info`, `debug`) |
| `--dev` | flag | Dry-run mode; no files are written |

---

### Extraction Modes

#### `direct` — single-stage extraction

The model receives the document text directly and is asked to extract entity pairs in a single step. This is the fastest mode and works well for clean, focused abstracts.

```
System prompt → Document text → [optional enrichments] → Extract triples
```

#### `nerrel` — two-stage NER + relation extraction

First, an NER call identifies relevant biological entities in the text. The entity list is then prepended to the relation-extraction prompt, narrowing the model's attention to plausible candidates.

```
System prompt → Document text
    → NER call → entity list
    → Relation extraction with entity context
```

`nerrel` is automatically activated when `--lookup`, `--all_nes_given`, `--true_nes_given`, or `--spacy_nes_given` is used.

---

### Prompt Strategies

#### `oneshot`

A single model call produces the final output. Suitable for most use cases and significantly faster than `stepwise`.

#### `stepwise`

A multi-turn refinement chain. The model first produces a broad extraction, then is guided through two additional filtering prompts to remove false positives (e.g., indirect signalling cascades for PPI, or PTM events for TF). Each turn receives the full conversation history so previous answers inform subsequent decisions.

The exact sequence of refinement prompts is target-specific:
- **PPI stepwise**: extract → filter for physical contact evidence → remove TF/gene interactions
- **TF stepwise**: extract → filter for direct transcriptional regulation → remove PTMs and protein interactions

---

### In-Context Examples

Static in-context examples demonstrate the expected output format and the boundary between true positives and false positives.

```bash
python extract.py … --examples negpos
```

| Value | Content |
|-------|---------|
| `pos` | Positive examples only (13 PPI / 5 TF) |
| `neg` | Negative examples only (6 PPI / 5 TF) — what *not* to extract |
| `negpos` | Both positive and negative examples |

Positive PPI examples include prototypic interactions such as p53–MDM2 (direct binding), AKT1–AKT1S1 (phosphorylation), and PIAS1–PNKP (SUMOylation). Negative examples show common false positives: co-expression, indirect pathway membership, structural similarity, and co-localisation.

---

### Dynamic Example Retrieval (DynEx)

Instead of fixed in-context examples, DynEx retrieves the *k* most semantically similar training-set documents (with their ground-truth triples) and injects them as examples at runtime. This adapts the few-shot context to each individual document.

```bash
python extract.py … --dynex_k 3
```

**How it works:**

1. At startup, `embed.py` builds (or loads) an HNSW vector index over the training corpus using the `mxbai-embed-large` embedding model served via Ollama.
2. For each new document, its embedding is computed and the index is queried for the `k × 2` nearest neighbours.
3. A diversity filter (cosine similarity threshold 0.8) selects up to *k* sufficiently distinct examples.
4. The selected examples are formatted and injected as a user turn before the extraction prompt.

The vector index is stored under `outputs/vectorstore/regulatome_{target}_idx.bin`. Build it once with `embed.py` if it does not yet exist.

---

### STRING Knowledge Lookup

When `--lookup` is enabled, the pipeline queries the [STRING](https://string-db.org/) protein interaction database before each extraction and appends known interaction partners as background knowledge.

```bash
python extract.py … --lookup
```

For each entity identified in the NER step, up to five known interaction partners (combined STRING score > 400) are retrieved and appended to the prompt:

```
BACKGROUND KNOWLEDGE:
TP53: Known PPIs: MDM2 (980), CDK2 (950), BRCA1 (910), …
```

Entity matching against STRING uses a combination of exact lookup, synonym expansion (from `synonyms.json`), and fuzzy full-text search (Whoosh, tolerance ≈ 2 characters).

---

### Ensemble Voting

Run *n* independent stochastic samples and retain only triples that appear in at least ⌈n/2⌉ samples (majority vote).

```bash
python extract.py … --ensemble 5 --ensemble_temp 0.8
```

This approach trades throughput for precision: noisy, hallucinated triples tend not to survive the majority filter, while consistently extracted true interactions are retained. The temperature (default 0.8) introduces the diversity necessary for effective voting.

---

### Tree of Thoughts (ToT)

ToT expands the extraction into multiple parallel reasoning paths, each driven by a different analytical strategy, and then combines the results.

```bash
python extract.py … --tot 3 --tot_strategy vote
```

**Workflow per document:**

1. **Strategy generation** — The LLM produces *n* distinct extraction strategies (e.g., *"focus on interaction verbs"*, *"look for co-IP evidence"*, *"scan PTM terminology"*).
2. **Path extraction** — Each strategy is applied independently to the document, producing a set of candidate triples.
3. **Path evaluation** — Each extracted triple is scored 1–10 for textual evidence quality.
4. **Combination** — Results from all paths are combined according to `--tot_strategy`:

| Strategy | Rule |
|----------|------|
| `vote`   | Keep triples appearing in ≥ ⌈n/2⌉ paths (default) |
| `best`   | Keep all triples from the highest-scoring path |
| `merge`  | Keep triples with score ≥ 8, or appearing in ≥ 2 paths, or (score ≥ 6 and in ≥ 2 paths) |

ToT is combinable with `--ensemble` for an outer layer of stochastic diversity on top of the multi-path reasoning.

---

## Output Format and Directory Layout

### Directory structure

Output paths are deterministically constructed from the run configuration by `paths.py`. The base path is:

```
{TRIPLES_ROOT}/{data}/{target}/{model}/{extractionmode}/{chattype}/{doclevel}/
```

Conditional subdirectories are appended in a fixed order:

| Condition | Subdirectory |
|-----------|-------------|
| `--doclevel chunks` | `{chunksize}/` |
| `--examples pos\|neg\|negpos` | `{examples}_ex/` |
| `--recall` | `recall/` |
| `--tot N` | `tot_nN_{strategy}/` |
| `--ensemble N` | `ensemble_nN_t{temp}/` |
| `--dynex_k K` | `dynex_kK/` |
| `--lookup` | `lookup/` |
| `--all_nes_given` | `all_nes_given/` |
| `--true_nes_given` | `true_nes_given/` |
| `--spacy_nes_given` | `spacy_nes_given/` |

**Example** — nerrel, oneshot, ToT with 3 paths (vote), ensemble of 5, DynEx k=3, with lookup:

```
outputs/triples/regulatome/ppi/llama33/nerrel/oneshot/docs/
  tot_n3_vote/ensemble_n5_t0.8/dynex_k3/lookup/
  triples.jsonl
```

### Record format

Each line in the `.jsonl` output is a self-contained JSON object:

```json
{
  "responses": [
    ["Protein1", "Protein2", "Protein3"],
    [
      {"head": "Protein1", "relation": "INTERACTS_WITH", "tail": "Protein2"},
      {"head": "Protein3", "relation": "INTERACTS_WITH", "tail": "Protein1"}
    ]
  ],
  "text": "Full source document text …",
  "filename": "/path/to/source.txt"
}
```

- `responses[0]` — entity list from the NER step (only present in `nerrel` mode)
- `responses[-1]` — final extracted triples after all refinement steps
- Each triple has `head`, `relation` (`INTERACTS_WITH` for PPI or `REGULATES` for TF), and `tail`

---

## SLURM / HPC Usage

All SLURM scripts in `slurm/` follow the same structure:

1. Start a local Ollama server on a dedicated port (default 11437).
2. Wait for the server to become ready.
3. Run one or more `extract.py` calls sequentially.
4. Kill the Ollama server on exit.

### Running the full regulatome PPI experiment matrix

`slurm/regu_ppi_matrix.sh` sweeps 15 configurations over two extraction modes (`direct`, `nerrel`) and eight prompt variants (`normal`, `neg`, `pos`, `negpos`, `dynex_k=3`, `lookup`, `ensemble=5`, `tot`):

```bash
sbatch slurm/regu_ppi_matrix.sh
```

Fixed parameters for the matrix: `--model llama33 --chattype oneshot --data regulatome --target ppi --doclevel docs --full_corpus`.

### Running a custom single experiment

```bash
sbatch --export=ALL slurm/regu_ppi_matrix.sh
# or submit inline:
sbatch --job-name=my_run --partition=gpu --gres=gpu:hopper:1 --mem=60G \
  --wrap="cd /path/to/scripts && \
    . ~/.venvs/test_linda/bin/activate && \
    OLLAMA_HOST=127.0.0.1:11437 python extract.py \
      --model llama33 --node local --port 37 \
      --data regulatome --target ppi \
      --extractionmode nerrel --chattype oneshot \
      --doclevel docs --full_corpus --force_new"
```

Key environment variables used in SLURM scripts:

| Variable | Value | Purpose |
|----------|-------|---------|
| `OLLAMA_HOST` | `127.0.0.1:11437` | Ollama endpoint for the main model |
| `OLLAMA_KEEP_ALIVE` | `4h` | Keep model weights in GPU memory |
| `OLLAMA_NUM_PARALLEL` | `1` | Serialise requests (avoids memory contention) |
| `OLLAMA_CONTEXT_LENGTH` | `80000` | Context window size |

---

## Synonym Generation

`synonyms.py` uses the LLM to generate alternative names, abbreviations, and aliases for every protein entity found in an extraction run. The resulting `synonyms.json` is used by `ppi_lookup.py` to improve fuzzy entity matching against the STRING database.

```bash
# Run via SLURM (recommended):
sbatch slurm/regu_ppi_synonyms.sh

# Or directly:
python synonyms.py \
  --model llama33 --node local --port 37 \
  --data regulatome --target ppi \
  --extractionmode direct --chattype oneshot --doclevel docs \
  --ext direct_normal_20260615_660834
```

The script automatically handles both the old (`.json`) and new (`.jsonl`) output formats. Output is written as `synonyms.json` alongside the source triples file.

---

## Fine-Tuning

LoRA-based supervised fine-tuning on extraction data:

```bash
python finetune.py --model llama31 --data regulatome --target ppi --save
```

Training conversations are derived from `dataset.py`, which pairs source documents with gold-standard triples formatted as chat turns. Checkpoints are written to `{OUTPUT_ROOT}/finetunedmodels/{model_id}_regulatome/`.

Fine-tuned model variants are registered in `clients.py` under the `*regu` suffix aliases (e.g., `llama33regu`, `llama31regu`) and can be used with any extraction flag combination.

---

## Datasets

| Corpus | Identifier | Description | Source |
|--------|-----------|-------------|--------|
| RegulaTome | `regulatome` | 1,591 PubMed abstracts with annotated PPI and TF relations | [Zenodo 10808330](https://zenodo.org/records/10808330) (CC BY 4.0) |
| BioRED | `biored` | Biomedical relation extraction benchmark | [BioRED](https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/) |
| 5 curated papers | `5curated` | Five manually curated cardiac signalling papers | Included under `data/5curated/` |
| Cardiac manuscripts | `cardio` | Broader collection of cardiac PDFs | Local, not distributed |

Place the RegulaTome files under `${LINDA_LLM_REGULATOME_ROOT}` (default: `../RegulaTome/`) or point the environment variable at your copy. The expected subdirectories are:

```
RegulaTome/
├── test_ppi_annotations/annotated_ppi_relations_dedup.txt
├── test_tf_annotations/annotated_tf_relations_dedup_new.txt
└── BIORED/…
```

---

## Model Hosting

The framework supports any **OpenAI-compatible** backend. Ollama is the primary local backend; Nebius (cloud) is also supported via `--nebius`.

### Supported model aliases

| Alias | Ollama model | HuggingFace (for finetuning) |
|-------|-------------|-------------------------------|
| `llama31` | `llama3.1-128k:8b` | `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` |
| `llama33` | `llama3.3:70b` | `unsloth/Llama-3.3-70B-Instruct-bnb-4bit` |
| `deepseek8b` | `deepseek-r1-128k:8b` | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` |
| `deepseek70b` | `deepseek-r1-128k:70b` | — |
| `gemma` | `gemma3:27b` | `unsloth/gemma-3-27b-it-unsloth-bnb-4bit` |
| `qwen3` | `qwen3:8b` | — |
| `qwen314` | `qwen3:14b` | — |
| `qwen330` | `qwen3:30b` | — |
| `qwen332` | `qwen3:32b` | — |
| `llama33regu` | `llama3.3:70b-regu_Q4_K_M` | fine-tuned on RegulaTome |
| `llama31regu` | `llama31regu-ollama` | fine-tuned on RegulaTome |

### Setting up Ollama

```bash
ollama serve &
ollama pull llama3.3:70b

# For DynEx (embedding model required):
ollama pull mxbai-embed-large
```

For multi-GPU clusters, map each node alias in `clients.py` to the appropriate IP address and Ollama port. The `--node local --port 37` combination (used in SLURM scripts) routes to `http://127.0.0.1:11437`, the in-job Ollama instance.
