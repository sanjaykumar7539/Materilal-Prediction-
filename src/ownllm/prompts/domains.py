"""Load domain profiles (system prompts + generation defaults) from disk."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


def _default_domains_root() -> Path:
    env = os.getenv("OWNLLM_DOMAINS_DIR", "").strip()
    if env:
        return Path(env)
    # Prefer repo-relative domains/ when running from source checkout
    here = Path(__file__).resolve()
    candidates = [
        here.parents[3] / "domains",  # src/ownllm/prompts -> repo root
        Path.cwd() / "domains",
        Path("/domains"),
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return Path.cwd() / "domains"


@dataclass
class DomainProfile:
    id: str
    name: str
    system_prompt: str
    chat_template: str = "chatml"
    max_new_tokens_default: int = 512
    temperature_default: float = 0.7
    top_p_default: float = 0.95
    top_k_default: int = 50
    description: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def format(self, user_message: str, context: Optional[str] = None) -> str:
        from .templates import format_prompt

        return format_prompt(
            user_message,
            system_prompt=self.system_prompt,
            template=self.chat_template,
            context=context,
        )


def list_domains(root: Optional[str | Path] = None) -> list[str]:
    base = Path(root) if root else _default_domains_root()
    if not base.is_dir():
        return []
    return sorted(
        p.name for p in base.iterdir() if p.is_dir() and (p / "domain.json").exists()
    )


def load_domain(domain_id: str, root: Optional[str | Path] = None) -> DomainProfile:
    base = Path(root) if root else _default_domains_root()
    folder = base / domain_id
    meta_path = folder / "domain.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Domain '{domain_id}' not found under {base}. "
            f"Available: {list_domains(base) or '(none)'}"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    prompt_file = meta.get("system_prompt_file", "SYSTEM_PROMPT.md")
    prompt_path = folder / prompt_file
    if not prompt_path.exists():
        raise FileNotFoundError(f"System prompt file missing: {prompt_path}")
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    return DomainProfile(
        id=meta.get("id", domain_id),
        name=meta.get("name", domain_id),
        system_prompt=system_prompt,
        chat_template=meta.get("chat_template", "chatml"),
        max_new_tokens_default=int(meta.get("max_new_tokens_default", 512)),
        temperature_default=float(meta.get("temperature_default", 0.7)),
        top_p_default=float(meta.get("top_p_default", 0.95)),
        top_k_default=int(meta.get("top_k_default", 50)),
        description=meta.get("description", ""),
        extra={k: v for k, v in meta.items() if k not in {
            "id", "name", "system_prompt_file", "chat_template",
            "max_new_tokens_default", "temperature_default",
            "top_p_default", "top_k_default", "description",
        }},
    )
