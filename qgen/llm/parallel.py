"""Independent batches of one pipeline step, run at the same time (krok 13b).

    run_batches(tasks, settings=..., on_done=...) -> results in the order of `tasks`

A task is one batch of a step (e.g. one LLM call with its repeated attempts). It must not
change the state of the pipeline: the caller merges the results in the main thread, in
the order of the batches, so the result is the same as when the batches ran one after
another.

QGEN_LLM_MAX_CONCURRENCY (settings.llm_max_concurrency):
    1     the tasks run in the calling thread, one after another (no threads)
    n > 1 at most n tasks run at once in a thread pool; the next task is started only
          when one has finished (so a failure or a cancel stops the batches not started)
The semaphore of qgen.llm.retry limits the LLM / embedding calls of all running tasks
together to the same number.

on_done(index, result) is called in the calling thread only, after every finished task
(in the order they finish). The progress callback of the pipeline (ProgressWriter) is
called from there, so it stays single-threaded.

Errors: when a task raises (LLMError, or any BaseException) or on_done raises (e.g.
JobCancelled / JobTimedOut from the progress callback), no new task is started, the
running tasks finish, and then the exception is raised: the task error with the lowest
index, else the on_done error. Nothing is returned, so there is never a partial result.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Callable, Optional, Sequence, TypeVar

from qgen.config import Settings

T = TypeVar("T")

THREAD_NAME_PREFIX = "qgen-llm"


def run_batches(
    tasks: Sequence[Callable[[], T]],
    *,
    settings: Settings,
    on_done: Optional[Callable[[int, T], None]] = None,
) -> list[T]:
    """Results of `tasks` in their order (see the module docstring)."""
    tasks = list(tasks)
    limit = max(1, int(settings.llm_max_concurrency))
    if limit == 1 or len(tasks) <= 1:
        results: list[T] = []
        for index, task in enumerate(tasks):
            result = task()
            results.append(result)
            if on_done is not None:
                on_done(index, result)
        return results
    return _run_in_threads(tasks, min(limit, len(tasks)), on_done)


def _run_in_threads(tasks: list[Callable[[], T]], workers: int, on_done) -> list[T]:
    results: dict[int, T] = {}
    task_errors: dict[int, BaseException] = {}
    stop_error: Optional[BaseException] = None
    running: dict[Future, int] = {}
    next_index = 0
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=THREAD_NAME_PREFIX)
    try:
        while next_index < len(tasks) and len(running) < workers:
            running[executor.submit(tasks[next_index])] = next_index
            next_index += 1
        while running:
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                index = running.pop(future)
                error = future.exception()
                if error is not None:
                    task_errors[index] = error
                    continue
                results[index] = future.result()
                if on_done is not None and not task_errors and stop_error is None:
                    try:
                        on_done(index, results[index])
                    except BaseException as callback_error:  # JobCancelled, JobTimedOut, ...
                        stop_error = callback_error
            stopped = bool(task_errors) or stop_error is not None
            while not stopped and next_index < len(tasks) and len(running) < workers:
                running[executor.submit(tasks[next_index])] = next_index
                next_index += 1
    finally:
        # Also on an exception in this thread (e.g. a job timeout): nothing new starts
        # (no task is queued in the executor), the running tasks finish.
        executor.shutdown(wait=True)
    if task_errors:
        raise task_errors[min(task_errors)]
    if stop_error is not None:
        raise stop_error
    return [results[index] for index in range(len(tasks))]
