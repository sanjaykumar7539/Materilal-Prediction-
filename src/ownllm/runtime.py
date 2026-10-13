"""Async inference runtime with concurrency gate, KV cache, and domain prompts."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Optional

import torch

from .inference.generate import apply_repetition_penalty, sample

log = logging.getLogger("ownllm.runtime")


class ModelRuntime:
    """Thread-safe (asyncio) generation wrapper around OwnLLM + tokenizer."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        device: Optional[str] = None,
        max_concurrency: int = 4,
        max_prompt_tokens: int = 4096,
        default_eos: Optional[str] = "<eos>",
        default_domain: Optional[str] = None,
        default_system_prompt: Optional[str] = None,
        chat_template: str = "chatml",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.gate = asyncio.Semaphore(max_concurrency)
        self.max_prompt_tokens = max_prompt_tokens
        self._eos_id = self._resolve_eos(default_eos)
        self.default_domain = default_domain
        self.default_system_prompt = default_system_prompt
        self.chat_template = chat_template
        self._domain_cache: dict[str, Any] = {}
        log.info(
            "ModelRuntime ready device=%s params=%s eos=%s domain=%s",
            self.device,
            getattr(model, "num_parameters", lambda: "?")(),
            self._eos_id,
            default_domain,
        )

    def _resolve_eos(self, token: Optional[str]) -> Optional[int]:
        if token is None:
            return None
        tok = self.tokenizer
        if hasattr(tok, "token_to_id"):
            tid = tok.token_to_id(token)
            if tid is not None:
                return int(tid)
        if hasattr(tok, "eos_token_id") and tok.eos_token_id is not None:
            return int(tok.eos_token_id)
        return None

    def encode(self, text: str) -> torch.Tensor:
        if hasattr(self.tokenizer, "encode"):
            enc = self.tokenizer.encode(text)
            ids = enc.ids if hasattr(enc, "ids") else list(enc)
        else:
            raise TypeError("tokenizer must provide encode()")
        if len(ids) > self.max_prompt_tokens:
            ids = ids[-self.max_prompt_tokens :]
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        if hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(ids)
        raise TypeError("tokenizer must provide decode()")

    def _resolve_system(
        self,
        system_prompt: Optional[str],
        domain: Optional[str],
    ) -> tuple[Optional[str], str, Optional[dict]]:
        """Return (system_text, template, domain_defaults)."""
        domain_id = domain or self.default_domain
        defaults: Optional[dict] = None
        if domain_id:
            if domain_id not in self._domain_cache:
                from ownllm.prompts import load_domain

                self._domain_cache[domain_id] = load_domain(domain_id)
            profile = self._domain_cache[domain_id]
            defaults = {
                "max_new_tokens": profile.max_new_tokens_default,
                "temperature": profile.temperature_default,
                "top_p": profile.top_p_default,
                "top_k": profile.top_k_default,
            }
            return (
                system_prompt if system_prompt is not None else profile.system_prompt,
                profile.chat_template,
                defaults,
            )
        return (
            system_prompt if system_prompt is not None else self.default_system_prompt,
            self.chat_template,
            defaults,
        )

    def build_prompt(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        domain: Optional[str] = None,
        context: Optional[str] = None,
    ) -> str:
        from ownllm.prompts import format_prompt

        sys_text, template, _ = self._resolve_system(system_prompt, domain)
        return format_prompt(
            user_message,
            system_prompt=sys_text,
            template=template,
            context=context,
        )

    async def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        system_prompt: Optional[str] = None,
        domain: Optional[str] = None,
        context: Optional[str] = None,
        apply_template: bool = True,
        repetition_penalty: float = 1.0,
        stop: Optional[list[str]] = None,
    ) -> str:
        _, _, defaults = self._resolve_system(system_prompt, domain)
        if defaults:
            if max_new_tokens == 256 and defaults.get("max_new_tokens"):
                max_new_tokens = int(defaults["max_new_tokens"])
            if temperature == 0.8 and defaults.get("temperature") is not None:
                temperature = float(defaults["temperature"])
            if top_p == 0.95 and defaults.get("top_p") is not None:
                top_p = float(defaults["top_p"])
            if top_k == 50 and defaults.get("top_k") is not None:
                top_k = int(defaults["top_k"])

        text = (
            self.build_prompt(prompt, system_prompt=system_prompt, domain=domain, context=context)
            if apply_template
            else prompt
        )
        stop = [s for s in (stop or []) if s]
        async with self.gate:
            ids = self.encode(text)
            caches = None
            out: list[int] = []
            seen: list[int] = ids.reshape(-1).tolist()
            text_out = ""
            with torch.inference_mode():
                logits, _, caches = self.model(ids, caches=caches)
                for _ in range(max_new_tokens):
                    step_logits = apply_repetition_penalty(
                        logits[:, -1], seen, repetition_penalty
                    )
                    nxt = sample(step_logits, temperature=temperature, top_k=top_k, top_p=top_p)
                    tid = int(nxt.item())
                    out.append(tid)
                    if self._eos_id is not None and tid == self._eos_id:
                        break
                    seen.append(tid)
                    if stop:
                        text_out += self.decode([tid])
                        hit = [text_out.find(s) for s in stop if s in text_out]
                        if hit:
                            text_out = text_out[: min(hit)]
                            return text_out
                    logits, _, caches = self.model(nxt, caches=caches)
            return text_out if stop else self.decode(out)

    async def stream_text(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        system_prompt: Optional[str] = None,
        domain: Optional[str] = None,
        context: Optional[str] = None,
        apply_template: bool = True,
        repetition_penalty: float = 1.0,
        stop: Optional[list[str]] = None,
    ) -> AsyncIterator[str]:
        _, _, defaults = self._resolve_system(system_prompt, domain)
        if defaults:
            if max_new_tokens == 256 and defaults.get("max_new_tokens"):
                max_new_tokens = int(defaults["max_new_tokens"])
            if temperature == 0.8 and defaults.get("temperature") is not None:
                temperature = float(defaults["temperature"])
            if top_p == 0.95 and defaults.get("top_p") is not None:
                top_p = float(defaults["top_p"])
            if top_k == 50 and defaults.get("top_k") is not None:
                top_k = int(defaults["top_k"])

        text = (
            self.build_prompt(prompt, system_prompt=system_prompt, domain=domain, context=context)
            if apply_template
            else prompt
        )
        stop = [s for s in (stop or []) if s]
        async with self.gate:
            ids = self.encode(text)
            caches = None
            seen: list[int] = ids.reshape(-1).tolist()
            text_out = ""
            with torch.inference_mode():
                logits, _, caches = self.model(ids, caches=caches)
                for _ in range(max_new_tokens):
                    step_logits = apply_repetition_penalty(
                        logits[:, -1], seen, repetition_penalty
                    )
                    nxt = sample(step_logits, temperature=temperature, top_k=top_k, top_p=top_p)
                    tid = int(nxt.item())
                    if self._eos_id is not None and tid == self._eos_id:
                        break
                    seen.append(tid)
                    if stop:
                        text_out += self.decode([tid])
                        hit = [text_out.find(s) for s in stop if s in text_out]
                        if hit:
                            cut = min(hit)
                            if cut > len(text_out) - len(self.decode([tid])):
                                # stop string started inside this piece: emit the safe part
                                yield text_out[len(text_out) - len(self.decode([tid])) : cut]
                            return
                        yield self.decode([tid])
                    else:
                        yield self.decode([tid])
                    logits, _, caches = self.model(nxt, caches=caches)
                    await asyncio.sleep(0)
