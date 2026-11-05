"""The mock image (Dockerfile.mock) copies only part of qgen/. Check that the app imports from it."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _copied_paths() -> list[str]:
    text = (ROOT / "Dockerfile.mock").read_text(encoding="utf-8")
    sources = []
    for match in re.finditer(r"^COPY\s+(\S+)\s+\S+\s*$", text, flags=re.MULTILINE):
        source = match.group(1).rstrip("/")
        if not source.endswith(".txt"):
            sources.append(source)
    return sources


def test_mock_image_file_set_imports_app(tmp_path):
    sources = _copied_paths()
    assert "service" in sources
    for source in sources:
        src = ROOT / source
        dst = tmp_path / source
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(src, dst)

    env = {k: v for k, v in os.environ.items() if not k.startswith("QGEN_")}
    env.update({"QGEN_BACKEND": "mock", "QGEN_API_KEYS": "test", "PYTHONPATH": str(tmp_path)})
    code = "import service.main as m; print(m.app.title)"
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert "MOCK" in result.stdout
