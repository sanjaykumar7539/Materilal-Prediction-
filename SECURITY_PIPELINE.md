# Local request and evidence pipeline

The supplied diagram is a design reference. This implementation uses the native OwnLLM and local components only.

The HTTP perimeter supports locally configured API keys with reader, writer and admin roles. With no keys configured, only direct loopback clients are accepted. Forwarded address headers are not trusted. Readers cannot submit corrections or override system prompts; metrics require admin permission. Configure OWNLLM_API_KEYS as a JSON object mapping strong keys to roles. The frontend must forward a valid key when authentication is enabled; never place secrets in frontend bundles.

Set OWNLLM_REQUIRE_TLS=1 when serving HTTPS directly through Uvicorn with a certificate and private key. Certificate provisioning and JWT verification are not provided. HTTP behind a TLS-terminating proxy will be refused by this setting unless the deployment explicitly and correctly configures its trusted proxy handling. Loopback HTTP remains the default for local development.

Requests are bounded by body bytes, the existing requests-per-minute limiter, conservative UTF-8 input byte plus output-token reservation, concurrency, pending queue size, queue wait and request duration. The extra token budget is per process and counts bytes as an upper-bound proxy for input tokens. The existing Redis request limiter remains available. These are not globally distributed concurrency or token limits. Cooperative request timeout cancels the async operation; it cannot forcibly interrupt a CPU/GPU kernel or background inference thread.

Input checks reject selected instruction-override patterns, key-shaped secrets and control characters. Output checks redact email addresses and withhold secrets. Streams are buffered before release, including across token boundaries. This intentionally delays the first visible token. These deterministic checks are limited heuristics, not a trained safety classifier, a complete PII scanner, or a guarantee against jailbreaks. Raw prompts, outputs and API keys are not written to audit logs; request path, hashed principal, status and duration are logged.

## Local evidence retrieval

Configure OWNLLM_KNOWLEDGE_INDEX to a local JSON array under OWNLLM_KNOWLEDGE_ROOT (default data), maximum 20 MB and 10,000 records. This is an operator-provisioned index; remote URLs, uploads and automatic ingestion are not accepted by the chat endpoint. Each record requires id and text and an explicit permission: public=true, owner equal to the authenticated principal hash, or roles containing an authorized role. Missing ACLs are denied. Source and score are optional metadata. Example:

```json
[{"id":"reviewed-note-1","text":"Reviewed source excerpt.","source":"Local datasheet, page 3","public":true}]
```

Send retrieve=true to /generate or /stream. The local BM25 index supplies evidence. Existing dense retrievers, local rerankers and graph adapters remain optional components of KnowledgeOrchestrator; this default bootstrap does not invent embeddings or graph facts. Secure retrieval filters permissions before fusion and after reranking. Secure calls re-read sources rather than reuse cached permissions, so ACL revocations are observed. Legacy orchestration without a principal remains available for trusted internal calls only. Multi-hop graph results without ACL metadata are denied in secured calls.

Evidence is placed in context as untrusted quoted data. Citation identifiers are checked against the retrieved map. In retrieval mode, output must consist of excerpts matching permitted evidence; unsupported sentences are refused. This conservative extractive check does not certify that source text is true or verify semantic entailment. /generate includes the citation map; /stream remains plain text. The strict materials profile route still uses independently reviewed records and its exact offline fallback, rather than bypassing those requirements through retrieved text.

Semantic caching and a trained local safety classifier are not enabled by default. The native model's RoPE runs on Q/K after projection; KV caches belong to attention decoding, rather than a separate post-sampling block as shown in the diagram. Model architecture and trained weights were not modified.

See tests/test_security_perimeter.py for authentication, size and token budgets, stream secret checks, access revocation, extractive grounding and timeout regression checks.
