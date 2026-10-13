# CampusMind OwnLLM v1.0

**Domain-neutral, offline-first, from-scratch decoder-only LLM engineering foundation.**

This repository intentionally contains **no domain-specific corpus** and **no pretrained LLM weights**. You bring your own text data, train from random initialization, and serve inference entirely offline.

## Architecture highlights

| Component | Details |
|-----------|---------|
| Model | RMSNorm · RoPE · GQA · SwiGLU · SDPA · weight-tied LM head |
| Training | FSDP/DDP helpers · atomic + sharded checkpoints · AMP (BF16/FP16) · grad accum/clip · activation checkpointing |
| Inference | KV-cached autoregressive generation · streaming · concurrency semaphore · priority scheduler |
| Alignment | SFT assistant-mask · DPO |
| Knowledge | BM25 + dense RAG · RRF · reranker · KAG multi-hop graph · CAG cache · evidence grounding/citations |
| API | FastAPI `/generate` + `/stream` · Redis or local rate limits · health/metrics |
| Offline | In-process socket guard · HF offline env flags · air-gap friendly Docker |

## Quick start

```bash
# Install
make install-dev

# Run tests
make test

# Prepare tokenizer + token shards from your .txt corpus
make prepare INPUT=data/raw OUTPUT=data/prepared

# Train from scratch (CPU or GPU)
make train SHARDS=data/prepared/shards/tokens_00000.bin STEPS=1000

# Serve API (set paths in .env)
cp .env.example .env
# Edit OWNLLM_MODEL_PATH and TOKENIZER_PATH
make run-api

# Or Docker + Redis
make docker-up
```

### Offline inference CLI

```bash
python scripts/infer_cli.py \
  --checkpoint runs/pretrain/final.pt \
  --tokenizer data/prepared/tokenizer/tokenizer.json \
  --prompt "Hello" \
  --max-new-tokens 64
```

## React chat UI

### Larger CampusMind configuration

To pretrain directly from a completed JSONL corpus containing `source`, `page`, `chunk_id` and `text`:

```powershell
python scripts/train_corpus_jsonl.py --input "C:\Users\Admin\Downloads\corpus.jsonl"
```

This runs the 18.88M-parameter configuration with a fresh 16,384-token tokenizer, source-group splits, and exact normalized chunk deduplication. The default 1,000-step run samples 256,000 token presentations rather than completing a full corpus pass. It evaluates sampled held-out token windows and exports weights, tokenizer, corpus audit, metrics, report and hashes under `runs/corpus_medium_v1/`. The 2,048-token model context is configured, while training sequences are 256 tokens. This is corpus pretraining; instruction tuning and material-answer evaluation remain separate stages. Existing chat checkpoints are preserved.

Continue training from saved weights and the exact same tokenizer into a new run:

```powershell
python scripts/train_corpus_jsonl.py --input data/drive_corpus/corpus.jsonl --base-export runs/corpus_medium_v1/export --out runs/corpus_medium_v2 --steps 1000 --lr 0.00015
```

Continuation checks that the model configuration and held-out source passages match the base export. It starts a new AdamW optimizer and learning-rate schedule; it does not resume the previous optimizer state. `--sampling-seed` controls training window selection independently of the fixed source split seed. Training records and exports identify the base checkpoint and additional token presentations.

The corpus reader also accepts JSON arrays of `source`, `page`, `chunk_id`, and `text` records. To continue on a different corpus, explicitly use `--new-corpus` with `--exclude-seen-input` pointing to the earlier corpus. Exact normalized previous chunks are excluded before the new source split; paraphrases and alternate editions may still overlap.

```powershell
python scripts/train_corpus_jsonl.py --input "C:/Users/Admin/Downloads/materials_science_corpus_master.json" --base-export runs/corpus_medium_v2/export --new-corpus --exclude-seen-input data/drive_corpus/corpus.jsonl --out runs/materials_master_v1 --steps 1000 --lr 0.00015
```

`configs/campusmind_medium.json` maps the proposed settings to this repository's native `ModelConfig`: vocabulary 16,384; hidden dimension 384; 8 layers; 12 attention heads; 4 KV heads; FFN dimension 1,024; maximum context 2,048. The tied-embedding model contains **18,880,896 trainable parameters**. The KV-head and FFN dimensions are explicit implementation choices for the fields omitted from the proposed snippet.

This project trains and runs its own local PyTorch `OwnLLM` model from random initialization. Training and inference use local weights and do not integrate external pretrained models or AI provider APIs. Changing model dimensions requires compatible weights and a matching tokenizer. The book trainer sets vocabulary from the actual tokenizer, so pass both `--config configs/campusmind_medium.json` and `--vocab 16384`.

For a separately saved run using the existing licensed text collection:

```powershell
python scripts/train_books.py --input data/open_materials_final --books data/open_materials/books --out runs/campusmind_medium_v1 --config configs/campusmind_medium.json --vocab 16384 --seq-len 2048 --batch 1 --steps 2000 --sft-epochs 6 --max-new-tokens 256
```

The larger model and longer sequences require more compute than the completed compact CPU runs. This command is documented for a future run; adding the configuration does not produce trained larger weights. Use a new output directory for each run.

### Fine-tune a supplied split QA/RAG dataset

For the ZIP containing `split_dataset/reviewed_qa/{train,validation,test}.jsonl` and corresponding `rag_knowledge` splits:

```powershell
python scripts/train_split_dataset.py --archive "C:\Users\Admin\Downloads\split_dataset-20261008T121232Z-1-001.zip"
```

The script continues the book-pretrained model with its matching tokenizer, removes duplicate normalized questions with test/validation precedence, converts each RAG field into a separate instruction example, selects the checkpoint using validation, and evaluates every held-out test example. It saves weights, tokenizer, predictions, metrics, provenance and hashes in `runs/split_dataset_v1/`. Use a new `--out` for later runs. Supplied verification flags remain metadata; this does not populate the reviewed materials profile collection. The ZIP has no declared license, and base textbook license records are saved separately.

### Strict materials policy

The materials domain renders engineering profiles from reviewed local records instead of sampling material values from the model. Profiles follow the supplied template: `### 1. General Overview`, then `### 2. Engineering Profile`. Each property uses a bold outer bullet and nested `Knowledge`, `Typical Value / Range`, `Unit / Scale`, and applicable `Formula / Equation` bullets. Sources and conditions stay inline within those fields. Missing materials or requested parameters return the exact verified-data fallback. Request context, system overrides and disabled templates cannot bypass this policy on `/generate` or `/stream`.

`domains/materials/verified_records.json` starts empty: downloaded books and trained weights are not automatically reviewed engineering records. Set `OWNLLM_MATERIALS_KB` to a reviewed collection. Each material needs `name`, optional `aliases`, `overview`, `overview_source`, `verified: true`, and `properties`. Each property needs `name`, `description`, `value`, `unit`, `source`, `conditions`, `applicable: true`, and `verified: true`. Add `formula` as reviewed LaTeX and set `formula_applicable: true` when an equation applies. An applicable missing equation excludes that property; properties without a governing equation omit the formula bullet. Invalid, unverified, irrelevant and placeholder entries are excluded. Verification flags represent a local reviewer decision, not automatic fact checking.

In the chat settings, select the materials domain and choose an engineering profile, review of provided data, or saved correction. Enter an exact material/grade and optionally a parameter. API requests accept `material`, `parameter` and `materials_action` (`profile`, `verify`, `feedback`). Verification provides an offline disclaimer, checks explicitly written scalar arithmetic, and flags missing references, units and conditions; it does not establish scientific accuracy. Corrections are saved locally as unverified feedback, considered in later verification requests for the same material, and never silently change model weights or reviewed facts. `OWNLLM_MATERIALS_FEEDBACK` selects the local SQLite file.

### Openly licensed training run

Download selected OpenStax editions and verify their embedded CC BY 4.0 declaration before extracting training text:

```powershell
python scripts/download_open_materials.py
python scripts/prepare_materials_curriculum.py --out data/open_materials_final --licensed-only
python scripts/train_books.py --input data/open_materials_final --books data/open_materials/books --out runs/open_materials_v1 --steps 2000 --sft-epochs 6 --max-new-tokens 256
python scripts/evaluate_materials_run.py --run runs/open_materials_v1
```

The downloader saves attribution, edition URLs, embedded license evidence and original-file hashes in `data/open_materials/`. Later OpenStax editions can have different license terms; admission is based on the specific downloaded PDF. The curriculum contains hypothetical formatting/selection exercises and checked arithmetic practice, rather than invented real material constants. `--licensed-only` excludes the earlier Drive instruction records with unestablished licenses. Test examples share task families; the benchmark does not establish general engineering competence. Exported weights, tokenizer, metrics and hashes are saved under the new run's `export/` directory.

The `frontend/` workspace provides a responsive React chat interface with streaming responses, browser-local conversation history, Markdown rendering, export, domain selection, and generation controls. Node.js 22 or newer is recommended.

Start the existing FastAPI service after configuring `OWNLLM_MODEL_PATH` and `TOKENIZER_PATH`:

```powershell
python -m uvicorn ownllm.api.service:app --app-dir src --host 127.0.0.1 --port 8000
```

Then start the frontend in a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite. Requests to `/api` are proxied to `http://127.0.0.1:8000`; set `OWNLLM_API_URL` before starting Vite to use a different backend. A healthy API without a loaded model shows a setup message and disables sending.

For the built UI with a Node.js server:

```powershell
cd frontend
npm run build
npm start
```

Open `http://127.0.0.1:3000`. The Node server serves the React build and proxies health, domains, and generation routes without requiring browser CORS configuration. `OWNLLM_API_URL`, `PORT`, and `HOST` configure the server; optional `OWNLLM_API_KEY` stays on the Node server. Binding to other interfaces requires your own authentication and deployment controls. Run `npm test` to validate request formatting and streaming error handling.

On Windows, `.\frontend\start.cmd` starts the built interface from the repository root. It uses the portable runtime downloaded to `.tools/` during setup or Node.js on PATH.

Conversation history is sent through the existing API's offline `context` field, so domain system prompts and templates continue to be applied by the backend. It is browser-local, without user accounts or cross-device sync. Stop cancels the browser request; backend generation cancellation depends on the runtime. No model weights or simulated assistant answers are included.

## Project layout

### Training document formats

### Train and evaluate a book collection

`scripts/train_books.py` trains a compact model from random initialization, applies assistant-only instruction tuning, selects checkpoints using validation loss, then evaluates held-out books, instruction examples, the supplied question CSV, and numerical problems. It saves weights, tokenizer, split audit, prediction samples, metrics, and a checksummed export folder. Its default 2.36M-parameter CPU configuration is a baseline, not a claim of engineering answer quality.

The input layout is `books/` for PDF, ZIP, TXT, EPUB, or DOCX documents and `dataset/` for `sft_train.jsonl`, `sft_validation.jsonl`, `sft_test.jsonl`, plus optional `evaluation.csv` and `numerical_problems.jsonl`. Instruction rows have `instruction`, optional `input`, and `output`. Books are split before tokenizer training; only the training partition and training instruction examples train the tokenizer. Exact normalized-text duplicates are removed before splitting. Near-duplicate editions and shared concepts remain possible and are disclosed in the report.

```powershell
python scripts/train_books.py --input data/drive_books --out runs/books_cpu --steps 1000 --sft-epochs 20
# Optional: prepare separately, then train from those prepared splits.
python scripts/train_books.py --input data/drive_books --out runs/books_cpu --prepare-only
python scripts/train_books.py --input data/drive_books --out runs/books_cpu --reuse-prepared
```

Use a different `--out` for every new run to preserve weights. An optional `--config` selects a larger ModelConfig; `--batch`, `--threads`, `--seq-len`, and `--eval-batches` control the run. Token-weighted loss/perplexity report the evaluated sample sizes. Generated-answer exact match, token F1, and numeric value matching do not establish factual accuracy. The provided instruction splits are audited for identical prompts; shared source concepts are recorded.

On Windows, after the `runs/books_cpu` run finishes, stop the old API process and run `.\scripts\serve_trained.cmd`. It loads the matching weights and tokenizer from `runs/books_cpu/export/` on port 8000 for the React chat UI. Other run folders can be served by setting `OWNLLM_MODEL_PATH` and `TOKENIZER_PATH` explicitly before starting Uvicorn.

`scripts/download_drive_books.py --manifest <manifest.json>` materializes a locally grounded public Drive file manifest. Install `python -m pip install -e ".[drive]"`. Each entry contains `id`, `name`, `size`, and a workspace-relative `relative_path`. Downloads verify expected sizes and record SHA256 hashes; cached files are reused. No external dataset scripts are executed.

Data preparation accepts a single file or a folder of `.txt`, `.text`, `.pdf`, `.epub`, `.docx`, and `.zip` files, including supported documents inside ZIPs. EPUB chapters follow the book's reading order. Originals remain unchanged; extracted text is temporary. Install updated dependencies with `python -m pip install -e .`.

```powershell
python scripts/prepare_tokens.py data/raw data/prepared --vocab 32000
# Or prepare one archive:
python scripts/prepare_tokens.py "C:/Documents/books.zip" data/prepared --vocab 32000
```

Unreadable files are reported and skipped. Scanned PDFs need OCR before preparation; password-protected PDFs need an unlocked copy. Microsoft Publisher `.pub` files must be exported to PDF or TXT (EPUB is supported). File uploads in the chat UI are not included in this training-input change.

Files are limited to 100 MiB each; each top-level archive has a shared 512 MiB decompression budget, 10,000 supported entries, and at most three nested ZIP levels. Archive contents are read in memory without writing member paths to disk. Larger collections can be split into smaller files or folders.

```
src/ownllm/
  model.py              # OwnLLM transformer
  runtime.py            # Async generation runtime
  offline.py            # Network guard
  api/                  # FastAPI service + bootstrap
  knowledge/            # RAG / KAG / CAG / fusion
  training/             # Trainer, checkpoints, SFT, DPO, FSDP
  inference/            # Sampling + scheduler
  data/                 # Memmap shards, cleaning, JSONL
  evaluation/           # Perplexity + promotion gate
scripts/                # CLI tools
configs/                # Model + large-training profiles
tests/                  # Smoke tests
```

## Version history (consolidated)

- **v1.0 (this release)** — Full code audit, readability pass, working model-load API, non-root Docker + healthcheck, KV-cache tests, offline CLI, expanded docs, stronger error handling.
- Prior iterations established the architecture, memmap shards, FSDP hooks, RAG/KAG/CAG orchestration, and release packaging.



## Materials Science domain (guidelines)

Your offline **Material Science & Advanced Engineering** system prompt is shipped under `domains/materials/`.

```bash
# Preview how the guidelines wrap a query (no model required)
python scripts/materials_cli.py --dry-run -q "Inconel 718"

# After training, generate with the materials system prompt + offline guard
python scripts/materials_cli.py --offline \
  --checkpoint runs/pretrain/final.pt \
  --tokenizer data/prepared/tokenizer/tokenizer.json \
  -q "Titanium Ti-6Al-4V"

# API: set default domain
export OWNLLM_DOMAIN=materials
export OWNLLM_DOMAINS_DIR=./domains
# POST /generate  { "prompt": "316L stainless", "domain": "materials" }
# GET  /domains
```

Rules enforced by the system prompt:
- Two sections: `### 1. General Overview` + `### 2. Engineering Profile`, with nested property bullets and reviewed equations where applicable
- Only relevant parameters (no N/A fillers)
- No hallucination — offline fallback sentence when data is missing
- External text can be checked for logical consistency only (not live DB verification)

## Generation controls (v1.1.1)

- `repetition_penalty` (1.0 = off, up to 2.0): presence penalty on already-generated tokens, applied in the decode loop.
- `stop` (up to 8 strings): halts generation at the first match and truncates output; streaming withholds the piece that completes the match.
- Both are available on `POST /generate` and `POST /stream`, and in `ModelRuntime.generate_text` / `stream_text`.
- Interactive multi-turn chat with history rendering:
  `make run-chat` or `python scripts/chat_cli.py --checkpoint models/ownllm.pt --tokenizer models/tokenizer.json [--domain materials]`

## Important notes

- **No pretrained weights.** Random init only. Quality depends on your corpus and compute.
- **GGUF / vLLM** are not claimed compatible until a custom architecture converter is validated for this tensor layout (RoPE, GQA, RMSNorm, SwiGLU).
- For a **hard offline guarantee**, combine `OWNLLM_OFFLINE=1` with OS/container egress blocking and a pre-populated wheelhouse.

See also: [ARCHITECTURE.md](ARCHITECTURE.md) · [DEPLOYMENT.md](DEPLOYMENT.md) · [OFFLINE.md](OFFLINE.md) · [SECURITY.md](SECURITY.md)
