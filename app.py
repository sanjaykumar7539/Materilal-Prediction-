"""Optional Gradio UI for an already-running OwnLLM API."""
from __future__ import annotations

import json
import os
import urllib.request

import gradio as gr

API = os.getenv("OWNLLM_API", "http://127.0.0.1:8000/generate")


def chat(message: str, history: list) -> str:
    body = json.dumps({"prompt": message, "max_new_tokens": 256}).encode()
    req = urllib.request.Request(
        API, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
            return data.get("text", str(data))
    except Exception as exc:
        return f"API unavailable: {exc}"


demo = gr.ChatInterface(
    fn=chat,
    title="CampusMind OwnLLM",
    description="Generic OwnLLM test console (requires running API with loaded model)",
)

if __name__ == "__main__":
    demo.launch()
