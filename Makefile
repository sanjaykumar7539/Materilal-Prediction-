.PHONY: help install install-dev install-ui test lint format precommit prepare train run-api run-ui run-chat docker-up docker-down clean

PYTHON ?= python
PIP ?= $(PYTHON) -m pip
INPUT ?= data/raw
OUTPUT ?= data/prepared
SHARDS ?= data/prepared/shards/tokens_00000.bin
CONFIG ?= configs/model.json
MODEL ?= models/ownllm.pt
TOKENIZER ?= models/tokenizer.json
STEPS ?= 1000
BATCH ?= 1

help:
	@echo "CampusMind OwnLLM v1.1"
	@echo "  make install       Install runtime dependencies"
	@echo "  make install-dev   Install development/test tools"
	@echo "  make install-ui    Install Gradio UI dependency"
	@echo "  make test          Run pytest"
	@echo "  make lint          Run Ruff checks"
	@echo "  make format        Format source/tests/scripts"
	@echo "  make prepare       Train tokenizer + build token shards"
	@echo "  make train         Start generic from-scratch pretraining"
	@echo "  make run-api       Start local FastAPI service"
	@echo "  make run-ui        Start local Gradio UI"
	@echo "  make docker-up     Start local Docker + Redis stack"
	@echo "  make docker-down   Stop local Docker stack"
	@echo "  make clean         Remove caches"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

install-ui:
	$(PIP) install -e ".[ui]"

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests scripts app.py docker_entrypoint.py

format:
	$(PYTHON) -m ruff format src tests scripts app.py docker_entrypoint.py
	$(PYTHON) -m ruff check --fix src tests scripts app.py docker_entrypoint.py

precommit:
	$(PYTHON) -m pre_commit run --all-files

prepare:
	$(PYTHON) scripts/prepare_tokens.py "$(INPUT)" "$(OUTPUT)"

train:
	$(PYTHON) scripts/train_from_scratch.py --config "$(CONFIG)" --shards "$(SHARDS)" --steps "$(STEPS)" --batch "$(BATCH)"

run-api:
	$(PYTHON) docker_entrypoint.py

run-ui:
	$(PYTHON) app.py

run-chat:
	$(PYTHON) scripts/chat_cli.py --checkpoint $(MODEL) --tokenizer $(TOKENIZER)

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	$(PYTHON) -c "import pathlib,shutil; [shutil.rmtree(p,ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
