"""tools/baseline.py: models from the settings or --app-models, start-up check, run info in
metrics.json and protection of the recordings baseline_v5/ (reference), baseline_v4/, baseline_v3/ and baseline/
(historical). No network, no pipeline run."""

import importlib.util
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from qgen.config import Settings

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def baseline():
    spec = importlib.util.spec_from_file_location("baseline_tool", REPO / "tools" / "baseline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _env_settings(**values):
    """Settings as if from .env, without any key (model_construct: no .env, no environment)."""
    return Settings.model_construct(**values)


def test_models_come_from_the_settings(baseline):
    env = _env_settings(llm_model="new-lite", llm_model_eval="new-flash", embed_model="emb-x")
    models = baseline.select_models(env, app_models=False)
    assert models == {
        "llm_model": "new-lite",
        "llm_model_lo": "new-lite",          # inherited
        "llm_model_questions": "new-lite",   # inherited
        "llm_model_eval": "new-flash",
        "embed_model": "emb-x",
    }


def test_app_models_keep_the_former_fixed_models(baseline):
    models = baseline.select_models(_env_settings(llm_model="new-lite"), app_models=True)
    assert models == baseline.APP_MODELS
    assert models["llm_model_lo"] == "gemini-2.5-flash-lite" and models["llm_model_eval"] == "gemini-2.5-flash"


def test_models_by_role_lists_every_role_and_embed(baseline):
    s = Settings.model_construct(llm_model="base", llm_model_questions="q", llm_model_eval="ev", embed_model="emb")
    roles = baseline.models_by_role(s)
    assert roles == {"lo": "base", "prereq": "base", "questions": "q", "eval": "ev",
                     "check": "base", "topics": "base", "split": "base", "embed": "emb"}


def test_reference_baseline_is_detected(baseline, tmp_path):
    assert baseline.is_reference_baseline(REPO / "baseline")
    assert baseline.is_reference_baseline(REPO / "tools" / ".." / "baseline")
    assert baseline.is_reference_baseline(REPO / "baseline_v3")
    assert baseline.is_reference_baseline(REPO / "baseline_v4")
    assert baseline.is_reference_baseline(REPO / "baseline_v5")
    assert not baseline.is_reference_baseline(REPO / "baseline_v2")
    assert not baseline.is_reference_baseline(tmp_path)


@pytest.mark.parametrize(
    "folder",
    ["baseline", "baseline_v3", "baseline_v4", "baseline_v5", "baseline_v5_single", "baseline_v5_topics",
     "baseline_v5_lessons3", "baseline_v5_syllabus_text"],
)
def test_out_protected_recording_is_refused_without_force(baseline, capsys, folder):
    with pytest.raises(SystemExit) as info:
        baseline.main([str(REPO / "tests" / "fixtures" / "eko.pdf"), "--out", str(REPO / folder)])
    assert info.value.code == 2
    assert "--force" in capsys.readouterr().err


def test_out_is_required(baseline):
    with pytest.raises(SystemExit):
        baseline.main([str(REPO / "tests" / "fixtures" / "eko.pdf")])


def test_config_is_checked_before_any_work(baseline, monkeypatch, tmp_path, capsys):
    """Missing key -> exit 2 after printing the models; no output folder, no client, no call."""
    pdf = tmp_path / "in.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(baseline, "get_settings", lambda: _env_settings(llm_model="m-lite", llm_model_eval=""))
    out = tmp_path / "run"
    assert baseline.main([str(pdf), "--out", str(out)]) == 2
    captured = capsys.readouterr()
    assert "eval       m-lite" in captured.out and "embed" in captured.out
    assert "GEMINI_API_KEY" in captured.err
    assert not out.exists()


def test_git_version(baseline, tmp_path):
    version = baseline.git_version()
    assert set(version) == {"commit", "dirty"}
    if version["commit"] is not None:
        assert len(version["commit"]) == 40 and isinstance(version["dirty"], bool)
    assert baseline.git_version(tmp_path) == {"commit": None, "dirty": None}  # not a repository


def test_metrics_contain_the_run_info(baseline, tmp_path):
    pdf = tmp_path / "in.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    settings = Settings.model_construct(llm_model="base", llm_model_eval="ev", embed_model="emb")
    run = baseline.run_info(settings, "settings", datetime(2026, 11, 4, 10, 0, tzinfo=timezone.utc),
                            {"commit": "a" * 40, "dirty": False})
    recorder = SimpleNamespace(counts=Counter(), errors=0)
    metrics = baseline.build_metrics(SimpleNamespace(), [str(pdf)], {}, None, None, [], [], {}, recorder,
                                     settings, run)
    assert metrics["run"] == {
        "started_at": "2026-11-04T10:00:00+00:00",
        "git_commit": "a" * 40,
        "git_dirty": False,
        "models_source": "settings",
        "llm_provider": "gemini",
        "embed_provider": "gemini",
        "llm_max_concurrency": 4,
        "models_by_role": {"lo": "base", "prereq": "base", "questions": "base", "eval": "ev",
                           "check": "base", "topics": "base", "split": "base", "embed": "emb"},
    }
    # Same keys as baseline_v5/metrics.json, with the models actually used.
    assert metrics["models"]["lo_generation"] == "base" and metrics["models"]["item_evaluation"] == "ev"
    assert metrics["models"]["lo_batch_size"] == 10


# --- --split / --base (step 16) -----------------------------------------------------------------


def test_base_needs_split(baseline, tmp_path):
    with pytest.raises(SystemExit):
        baseline.main([str(REPO / "tests" / "fixtures" / "eko.pdf"), "--base", "baseline_v5", "--out", str(tmp_path)])


def test_base_run_records_only_the_split_calls(baseline, monkeypatch, tmp_path):
    """--base baseline_v5: the pipeline is replayed, only the lesson titles call the (fake)
    API and land in llm_calls.jsonl; units.json, report.json and metrics.json are written."""
    import json

    from tests.baseline_data import BASELINE_DIR, require_baseline
    from tests.fakes import FakeEmbedder, FakeLLM, check_answer

    require_baseline()
    fake = FakeLLM(default=lambda prompt, model: check_answer(prompt) or '{"items": []}')
    monkeypatch.setattr(baseline, "get_settings", lambda: Settings.model_construct())
    monkeypatch.setattr(baseline.llm_factory, "check_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(baseline.llm_factory, "get_llm", lambda role, settings: fake)
    monkeypatch.setattr(baseline.llm_factory, "get_embedder", lambda settings: FakeEmbedder())
    out = tmp_path / "split"
    code = baseline.main([str(REPO / "tests" / "fixtures" / "eko.pdf"), "--base", str(BASELINE_DIR),
                          "--split", '{"mode": "lessons", "lesson_count": 3}', "--out", str(out)])
    assert code == 0
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["base"]["dir"] == "baseline_v5" and metrics["base"]["unused_base_calls"] == 0
    assert metrics["split"]["options"]["lesson_count"] == 3 and metrics["split"]["lessons"] == 3
    assert metrics["llm_calls"]["by_method"] == {"generate_content": 1}
    assert len((out / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    units = json.loads((out / "units.json").read_text(encoding="utf-8"))
    assert [u["title"] for u in units["units"]] == ["Hodina 1 – názov", "Hodina 2 – názov", "Hodina 3 – názov"]
    assert (out / "report.json").is_file()
