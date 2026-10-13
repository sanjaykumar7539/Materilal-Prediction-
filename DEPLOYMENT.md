# Deployment

## Local

1. Copy `.env.example` → `.env`.
2. Place trained checkpoint and tokenizer under `models/` (gitignored).
3. Set:
   ```
   OWNLLM_MODEL_PATH=./models/ownllm.pt
   TOKENIZER_PATH=./models/tokenizer.json
   DEVICE=cpu   # or cuda
   ```
4. `make run-api` or `python docker_entrypoint.py`.

## Docker Compose

```bash
cp .env.example .env
# point OWNLLM_MODEL_PATH to /models/...
docker compose up --build
```

- Redis for distributed rate limits / optional CAG
- Model volume mounted read-only
- Non-root container user (`ownllm`, uid 10001)
- Healthcheck on `/health`

## Export

Native PyTorch checkpoint is the canonical format:

```bash
python scripts/export_model.py \
  --config configs/model.json \
  --checkpoint runs/pretrain/final.pt \
  --format native \
  --output artifacts/model.pt
```

ONNX (logits-only, no KV cache):

```bash
python scripts/export_model.py ... --format onnx --output artifacts/model.onnx
```

## Production checklist

- [ ] TLS termination at reverse proxy / ingress
- [ ] API keys (`x-api-key`) + Redis rate limits
- [ ] Resource quotas and load tests
- [ ] Read-only model volumes
- [ ] Centralized structured logs
- [ ] Dependency and container scanning
- [ ] Offline egress policy if air-gapped
