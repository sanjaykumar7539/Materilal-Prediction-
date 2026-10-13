from pathlib import Path
def test_console_entry_points_to_packaged_module():
    text=(Path(__file__).parents[1]/"pyproject.toml").read_text()
    assert 'ownllm-infer = "ownllm.cli:main"' in text
def test_make_run_chat_is_phony():
    text=(Path(__file__).parents[1]/"Makefile").read_text().splitlines()[0]
    assert "run-chat" in text
