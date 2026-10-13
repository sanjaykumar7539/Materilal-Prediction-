# Security baseline

- Never commit `.env`, credentials, private corpora, model checkpoints, or signing keys.
- Put TLS and authentication at the ingress; keep Redis private (no public bind).
- Bound prompt/output sizes and concurrency (`MAX_CONCURRENCY`, request Field limits).
- Apply authorization **before** private RAG retrieval, not after generation.
- Treat retrieved documents and user prompts as untrusted input.
- Rotate exposed secrets immediately.
- Enable dependency and container scanning in the hosting repository.
- Run containers as non-root (default image user `ownllm`).
- Prefer read-only mounts for model weights.
- Rate-limit by API key or client IP; fail closed on Redis errors by falling back to local limits.
