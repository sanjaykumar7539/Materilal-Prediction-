"""Production app factory: attach a ModelRuntime and optional Redis client."""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .limits import DistributedRateLimiter

log = logging.getLogger("ownllm.api")


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=32000)
    max_new_tokens: int = Field(256, ge=1, le=4096)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: int = Field(50, ge=0, le=500)
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    domain: Optional[str] = None
    system_prompt: Optional[str] = Field(default=None, max_length=100000)
    context: Optional[str] = Field(default=None, max_length=50000)
    apply_template: bool = True
    repetition_penalty: float = Field(1.0, ge=1.0, le=2.0)
    stop: Optional[list[str]] = Field(default=None, max_length=8)


def create_app(runtime: Any, redis_client: Optional[Any] = None) -> FastAPI:
    """Build a FastAPI app bound to a live ModelRuntime."""
    app = FastAPI(
        title="CampusMind OwnLLM",
        version="1.1.2",
        description="Offline-first OwnLLM inference API with domain system prompts",
    )
    limiter = DistributedRateLimiter(
        redis_client,
        limit=int(os.getenv("RPM", "30")),
        window=int(os.getenv("RATE_WINDOW", "60")),
    )
    app.state.runtime = runtime
    app.state.limiter = limiter
    default_domain = getattr(runtime, "default_domain", None) or os.getenv("OWNLLM_DOMAIN") or None

    @app.get("/health")
    async def health():
        return {
            "ok": True,
            "model_loaded": runtime is not None,
            "device": getattr(runtime, "device", None),
            "default_domain": default_domain,
        }

    @app.get("/domains")
    async def domains():
        from ownllm.prompts import list_domains

        return {"domains": list_domains()}

    @app.get("/metrics")
    async def metrics_endpoint():
        from ownllm.observability import metrics

        return metrics.snapshot()

    def _client_key(request: Request) -> str:
        return request.headers.get("x-api-key") or (
            request.client.host if request.client else "unknown"
        )

    @app.post("/generate")
    async def generate(req: GenerateRequest, request: Request):
        ok, retry = await limiter.allow(_client_key(request))
        if not ok:
            raise HTTPException(
                status_code=429,
                detail="rate limit exceeded",
                headers={"Retry-After": str(retry)},
            )
        from ownllm.observability import metrics

        metrics.inc("requests")
        try:
            text = await runtime.generate_text(
                req.prompt,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                system_prompt=req.system_prompt,
                domain=req.domain or default_domain,
                context=req.context,
                apply_template=req.apply_template,
            )
            return {"text": text, "domain": req.domain or default_domain}
        except FileNotFoundError as exc:
            metrics.inc("errors")
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            log.exception("generate failed")
            metrics.inc("errors")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/stream")
    async def stream(req: GenerateRequest, request: Request):
        ok, retry = await limiter.allow(_client_key(request))
        if not ok:
            raise HTTPException(
                status_code=429,
                detail="rate limit exceeded",
                headers={"Retry-After": str(retry)},
            )
        from ownllm.observability import metrics

        metrics.inc("requests")

        async def gen():
            try:
                async for piece in runtime.stream_text(
                    req.prompt,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    top_k=req.top_k,
                    top_p=req.top_p,
                    system_prompt=req.system_prompt,
                    domain=req.domain or default_domain,
                    context=req.context,
                    apply_template=req.apply_template,
                ):
                    yield piece
            except Exception as exc:
                log.exception("stream failed")
                metrics.inc("errors")
                yield f"\n[error] {exc}\n"

        return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")

    return app
