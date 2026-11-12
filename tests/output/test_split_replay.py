"""Replay of the split step (step 16) on eko.pdf: pipeline from baseline_v5, split and output
from the short recordings baseline_v5_single / _topics / _lessons3 / _syllabus_text.

The result (units.json) and the report (report.json, without timings) must equal the
recording, and every recorded split call must be used. Skipped until the recordings exist
locally (tools/baseline.py --base baseline_v5 --split ...).
"""

import pytest
from pydantic import TypeAdapter

from qgen.contract.options import SplitOptions
from qgen.output import build_output, check_output
from qgen.pipeline import run_pipeline
from tests.baseline_data import (
    SPLIT_RECORDINGS,
    baseline_settings,
    load_json,
    load_split_json,
    require_split_recording,
    split_replay_client,
)


@pytest.mark.parametrize("name", SPLIT_RECORDINGS)
def test_split_replay_equals_the_recording(name):
    require_split_recording(name)
    metrics = load_split_json(name, "metrics.json")
    split = TypeAdapter(SplitOptions).validate_python(metrics["split"]["options"])
    replay = split_replay_client(name)
    settings = baseline_settings()
    result = run_pipeline(load_json("segments.json"), llm=replay, embedder=replay, settings=settings)
    output = build_output(
        result,
        input_names=[i["name"] for i in metrics["base"]["inputs"]],
        timings={},
        settings=settings,
        split=split,
        llm=replay,
        embedder=replay,
    )
    check_output(output.result, output.report)

    assert output.result.model_dump(mode="json") == load_split_json(name, "units.json")
    report = output.report.model_dump(mode="json")
    recorded = load_split_json(name, "report.json")
    report.pop("timings"), recorded.pop("timings")
    assert report == recorded
    assert len(output.result.units) == metrics["split"]["lessons"]
    assert replay.unused_records(recording=1) == []
