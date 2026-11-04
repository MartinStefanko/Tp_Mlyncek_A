"""Job queue interface. The RQ implementation comes in step 10; tests use InMemoryJobQueue."""

from typing import Protocol


class JobQueue(Protocol):
    def enqueue(self, job_id: str) -> None:
        """Put the job into the queue for a worker."""

    def contains(self, job_id: str) -> bool:
        """True if the job waits in the queue or runs on a LIVE worker.

        A job whose worker died (it stayed in the registry of started jobs) does not count –
        otherwise recovery would never bring it back.
        """


class InMemoryJobQueue:
    """FIFO queue in memory, with a set of job ids that run on a live worker."""

    def __init__(self) -> None:
        self.waiting: list[str] = []
        self.running: set[str] = set()

    def enqueue(self, job_id: str) -> None:
        self.waiting.append(job_id)

    def contains(self, job_id: str) -> bool:
        return job_id in self.waiting or job_id in self.running
