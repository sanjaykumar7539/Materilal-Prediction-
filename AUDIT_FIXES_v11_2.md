# v11.2 Audit Fixes

Verified before fixes: Python compile passed; 18/18 existing tests passed.

Fixed integration defects:
1. `ownllm-infer` console entry referenced non-packaged `scripts.infer_cli`; moved implementation to `ownllm.cli`.
2. Gradient accumulation CLI value was dividing loss without accumulating that many microbatches; training loop and trainer normalization corrected.
3. `/generate` accepted `repetition_penalty` and `stop` but did not pass them to runtime; corrected in service and app factory.
4. `make run-chat` was missing from `.PHONY`.
5. Docker Compose now overrides model/tokenizer/domain paths to container paths.
6. Offline socket guard can optionally allow private Docker-network addresses with `OWNLLM_OFFLINE_ALLOW_PRIVATE=1`; public outbound connections remain blocked.
7. Package/API version aligned to 1.1.2.
8. Added regression tests for packaging metadata and gradient accumulation.

Environment note: dependency installation was not performed during this audit because the audit runtime has no package-index network access. Existing tests that do not require missing optional packages were executed.
