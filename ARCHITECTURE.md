# Architecture

## Model

The implemented request perimeter, permission-filtered evidence retrieval, output checks and deployment settings are documented in [SECURITY_PIPELINE.md](SECURITY_PIPELINE.md). The security perimeter wraps both generation routes, including streamed responses, before returning text to the client.

Training and inference use the project's own local `OwnLLM`, initialized from scratch and continued from locally saved checkpoints. No external pretrained model or AI provider API is integrated. The local HTTP backend connects the frontend to this model.

Decoder-only Transformer (`OwnLLM`):

- Token embedding + weight-tied LM head
- Per-layer: RMSNorm → GQA Attention (RoPE) → residual → RMSNorm → SwiGLU FFN → residual
- Scaled Dot-Product Attention (`torch.nn.functional.scaled_dot_product_attention`)
- Optional activation checkpointing for memory-constrained training
- KV cache for autoregressive decode

Default small config (`configs/model.json`): 768-d, 12 layers, 12 heads / 4 KV heads, 2K context.  
Large training profile (`configs/large_training.json`): 1024-d, 24 layers, 8K context, grad-accum 32, BF16.

## Training stack

- `Trainer` — AMP, gradient accumulation, clipping
- `MemmapTokenDataset` — next-token windows over binary shards larger than RAM
- Atomic checkpoint + optional `torch.distributed.checkpoint` (FSDP)
- SFT (assistant-token mask) and DPO preference step
- Run-state recovery JSON for resumable jobs

## Knowledge layer

```
query → CAG cache → dense + BM25 RAG → RRF → rerank
      → KAG multi-hop graph → grounded context + citations
      → OwnLLM → response
```

- `LocalVectorStore` — numpy cosine search (mmap)
- `BM25Retriever` / `DenseRetriever`
- `KnowledgeGraph` (in-memory) + `PersistentKAG` (SQLite)
- `SemanticCache` — embedding-similarity TTL cache
- `KnowledgeOrchestrator` — end-to-end fusion

## Serving

```
API key / IP → rate limiter (Redis or local) → concurrency semaphore
            → ModelRuntime (KV-cached generate / stream)
```

- `create_app(runtime, redis_client)` for custom bootstrap
- Default `ownllm.api.service:app` auto-loads checkpoint when env paths are set
- `/health`, `/metrics`, `/generate`, `/stream`

## Offline posture

`enforce_offline()` blocks non-loopback sockets inside the process and sets HF offline env vars.  
Hard guarantee requires host/network egress denial and offline wheel installation.

## Materials V3 reporting upgrade

See [MATERIALS_V3_ROADMAP.md](MATERIALS_V3_ROADMAP.md) for the inspected capabilities, implemented query/reporting changes and remaining scientific validation gates. Materials responses use reviewed local records and deterministic routing before native generation.
