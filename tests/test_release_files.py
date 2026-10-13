from pathlib import Path


def test_release_files_present():
    root = Path(__file__).parents[1]
    for name in [
        ".gitignore",
        ".env.example",
        "Dockerfile",
        "docker-compose.yml",
        "pyproject.toml",
        "README.md",
        "SECURITY.md",
        "OFFLINE.md",
        "ARCHITECTURE.md",
        "DEPLOYMENT.md",
    ]:
        assert (root / name).exists(), name
