"""Job events for webhooks: the interface used by the API, the worker and recovery.

The real implementation is service.webhooks.rq_delivery.RqWebhookNotifier (queue
"qgen-webhooks", served by the webhook worker). `service/jobs` depends only on this
Protocol, not on rq or the webhook modules.

Rule for the callers: write job.json first, then call `notify` with the record returned by
that write. So the body of the webhook is exactly the state that GET /v1/jobs/{id} returns.
`notify` never raises: a delivery problem never changes the state of a job.
"""

from dataclasses import dataclass, field
from typing import Protocol

from qgen.contract import WebhookEvent
from service.jobs.models import JobRecord


class JobEvents(Protocol):
    def notify(self, event: WebhookEvent, record: JobRecord) -> None:
        """Queue the webhook `event` with the state of `record` (nothing without callback_url)."""

    def has_live_worker(self) -> bool:
        """True if at least one live worker delivers webhooks (GET /health)."""


@dataclass
class InMemoryJobEvents:
    """Records the events (tests of the API and recovery)."""

    sent: list[tuple[WebhookEvent, JobRecord]] = field(default_factory=list)
    worker_alive: bool = True

    def notify(self, event: WebhookEvent, record: JobRecord) -> None:
        if record.options.callback_url is not None:
            self.sent.append((event, record.model_copy(deep=True)))

    def has_live_worker(self) -> bool:
        return self.worker_alive

    def events(self, job_id: str | None = None) -> list[str]:
        return [event for event, record in self.sent if job_id is None or record.job_id == job_id]
