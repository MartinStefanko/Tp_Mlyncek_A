"""Progress reporting of the pipeline: progress_callback(stage, done, total).

Stage names are the pipeline values of JobStage in the contract (kap. 4). A step reports
(stage, 0, total) when it starts and (stage, done, total) after every batch; the last
call of a step has done == total. Without a callback nothing happens.
"""

from typing import Callable, Optional

ProgressCallback = Callable[[str, int, int], None]

STAGE_EXTRACT = "extract"
STAGE_LANGUAGE = "language"
STAGE_LEARNING_OBJECTS = "learning_objects"
STAGE_LO_EVALUATION = "lo_evaluation"
STAGE_QUESTIONS = "questions"
STAGE_Q_EVALUATION = "q_evaluation"
# After the pipeline (qgen.output.build_output): split into lessons, then the output.
STAGE_SPLIT = "split"
STAGE_OUTPUT = "output"

PIPELINE_STAGES = (
    STAGE_EXTRACT,
    STAGE_LANGUAGE,
    STAGE_LEARNING_OBJECTS,
    STAGE_LO_EVALUATION,
    STAGE_QUESTIONS,
    STAGE_Q_EVALUATION,
)


def report_progress(callback: Optional[ProgressCallback], stage: str, done: int, total: int) -> None:
    if callback is not None:
        callback(stage, done, total)


class SubProgress:
    """Callback for a sub-step whose batches are one part of a bigger step.

    The sub-step reports its own (stage, done, sub_total). This object forwards
    (stage, before + done, before + sub_total + after): `before` units of the step are
    already done, about `after` units follow. `sub_total` keeps the last total the
    sub-step reported, so the next part can start after it.
    """

    def __init__(self, callback: Optional[ProgressCallback], before: int = 0, after: int = 0):
        self.callback = callback
        self.before = before
        self.after = after
        self.sub_total = 0

    def __call__(self, stage: str, done: int, sub_total: int) -> None:
        self.sub_total = sub_total
        report_progress(self.callback, stage, self.before + done, self.before + sub_total + self.after)


def counting_progress(callback: Optional[ProgressCallback], stage: str, total: int) -> Callable[[int, object], None]:
    """on_done of qgen.llm.parallel.run_batches: (stage, finished batches, total) after
    every finished batch, whichever batch it was (called in the main thread only)."""
    finished = 0

    def on_done(_index: int, _result: object) -> None:
        nonlocal finished
        finished += 1
        report_progress(callback, stage, finished, total)

    return on_done


def batch_count(size: int, batch_size: int) -> int:
    return (size + batch_size - 1) // batch_size if size > 0 else 0
