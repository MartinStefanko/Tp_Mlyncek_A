"""In-memory mock of the job backend (for Laravel integration before the pipeline exists)."""

from service.mock.backend import MockBackend, MockFixtures

__all__ = ["MockBackend", "MockFixtures"]
