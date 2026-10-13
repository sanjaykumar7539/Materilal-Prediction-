"""Chat templates for OwnLLM (offline, no external tokenizer chat APIs required)."""
from __future__ import annotations

from typing import Optional


def apply_chat_template(
    user_message: str,
    system_prompt: Optional[str] = None,
    template: str = "chatml",
) -> str:
    """Build a single prompt string for autoregressive generation.

    Templates:
      - chatml: ChatML-style tags
      - plain: system + user concatenated with clear headers
      - instruct: Alpaca-style instruction block
    """
    system = (system_prompt or "").strip()
    user = user_message.strip()

    if template == "chatml":
        parts = []
        if system:
            parts.append(f"<|im_start|>system\n{system}<|im_end|>")
        parts.append(f"<|im_start|>user\n{user}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    if template == "instruct":
        if system:
            return (
                f"### System\n{system}\n\n"
                f"### Instruction\n{user}\n\n"
                f"### Response\n"
            )
        return f"### Instruction\n{user}\n\n### Response\n"

    # plain (default fallback)
    if system:
        return f"[SYSTEM]\n{system}\n\n[USER]\n{user}\n\n[ASSISTANT]\n"
    return f"[USER]\n{user}\n\n[ASSISTANT]\n"


def format_prompt(
    user_message: str,
    system_prompt: Optional[str] = None,
    template: str = "chatml",
    context: Optional[str] = None,
) -> str:
    """Optionally inject retrieved context before the user turn."""
    if context and context.strip():
        enriched = (
            f"Use the following offline knowledge context when relevant. "
            f"Do not invent facts beyond it.\n\n"
            f"--- CONTEXT ---\n{context.strip()}\n--- END CONTEXT ---\n\n"
            f"{user_message.strip()}"
        )
        return apply_chat_template(enriched, system_prompt=system_prompt, template=template)
    return apply_chat_template(user_message, system_prompt=system_prompt, template=template)
