"""Calibration of QGEN_QUESTION_DEDUP_THRESHOLD on a recording (step 13).

Reads <recording>/questions_all.json and the recorded embeddings in llm_calls.jsonl (no
API call). For every pair of questions of the same section it computes the cosine
similarity of the dedup texts (qgen.questions.dedup.dedup_text) and prints the most
similar pairs and the quantiles. Pairs whose text was not embedded in the recording are
skipped (the pipeline embeds the dedup text of every valid question).

Usage (from the repo root):
    venv/Scripts/python.exe tools/dedup_report.py baseline_v5 [--top 30]
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from qgen.questions.dedup import cosine, dedup_text  # noqa: E402


def recorded_vectors(calls_path: Path) -> dict:
    vectors = {}
    with open(calls_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("method") != "embed_content" or "output" not in record:
                continue
            for text, vector in zip(record["contents"], record["output"].get("embeddings", [])):
                vectors[text] = vector
    return vectors


def section_pairs(questions: list[dict], vectors: dict) -> list[tuple[float, dict, dict]]:
    pairs = []
    by_section: dict = {}
    for question in questions:
        by_section.setdefault(question.get("lo_id"), []).append(question)
    for group in by_section.values():
        for a, b in itertools.combinations(group, 2):
            va, vb = vectors.get(dedup_text(a)), vectors.get(dedup_text(b))
            if va is not None and vb is not None:
                pairs.append((cosine(va, vb), a, b))
    pairs.sort(key=lambda pair: pair[0], reverse=True)
    return pairs


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording", help="Recording folder, e.g. baseline_v5")
    parser.add_argument("--top", type=int, default=30, help="Number of most similar pairs to print.")
    args = parser.parse_args(argv)

    folder = Path(args.recording)
    questions = json.loads((folder / "questions_all.json").read_text(encoding="utf-8"))
    pairs = section_pairs(questions, recorded_vectors(folder / "llm_calls.jsonl"))
    if not pairs:
        print("No pair of questions of the same section with recorded embeddings.")
        return 1
    values = [score for score, _, _ in pairs]
    print(f"{len(pairs)} pairs within sections")
    for q in (0.5, 0.75, 0.9, 0.95, 0.99):
        print(f"  quantile {q:.2f}: {quantile(values, q):.4f}")
    print(f"  max: {values[0]:.4f}\n")
    for score, a, b in pairs[: args.top]:
        mark = " (dropped as duplicate)" if b.get("drop_reason") == "duplicate" or a.get("drop_reason") == "duplicate" else ""
        print(f"{score:.4f}  section {a.get('lo_id')}{mark}")
        print(f"    A [{a.get('type')}] {' '.join(str(a.get('description', '')).split())[:140]}")
        print(f"    B [{b.get('type')}] {' '.join(str(b.get('description', '')).split())[:140]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
