from pathlib import Path

from ownllm.prompts import apply_chat_template, format_prompt, list_domains, load_domain


def test_materials_domain_loads():
    # domains/ lives at repo root relative to package
    root = Path(__file__).resolve().parents[1] / "domains"
    assert "materials" in list_domains(root)
    profile = load_domain("materials", root=root)
    assert "Material Science" in profile.name or "Materials" in profile.name
    assert "### 1. General Overview" in profile.system_prompt
    assert "Engineering Profile" in profile.system_prompt or "ENGINEERING PROFILE" in profile.system_prompt
    assert "NO FILLERS" in profile.system_prompt or "N/A" in profile.system_prompt
    formatted = profile.format("Titanium Ti-6Al-4V")
    assert "Titanium" in formatted
    assert "system" in formatted.lower() or "SYSTEM" in formatted or "im_start" in formatted


def test_chat_templates():
    s = apply_chat_template("hello", system_prompt="be helpful", template="chatml")
    assert "hello" in s and "be helpful" in s
    s2 = format_prompt("q", system_prompt="sys", context="ctx block", template="plain")
    assert "ctx block" in s2 and "q" in s2
