"""Detection of an answer that degenerated into whitespace.

A local model in JSON mode (e.g. Gemma on vLLM with response_format json_object) can get
stuck and write only spaces and newlines until the output token limit: the constrained
decoding always allows whitespace. Measured on 01.pdf: 32 768 output tokens, 99.9 %
whitespace after a broken first question. A normal JSON answer has short whitespace runs
(indentation), never hundreds of characters in a row.

    trailing_whitespace_run  length of the whitespace run at the end of a text
    WhitespaceRunGuard       the same over a stream of chunks (openai_compatible streaming)

The limit is QGEN_LLM_MAX_WHITESPACE_RUN. Adapters raise DegenerateOutputError.
"""

from __future__ import annotations

from qgen.llm.errors import DegenerateOutputError, OutputTruncatedError, ProviderError

DEFAULT_MAX_WHITESPACE_RUN = 500


def trailing_whitespace_run(text: str) -> int:
    return len(text) - len(text.rstrip())


def cut_off_error(provider: str, reason: str, text: str, limit: int) -> ProviderError:
    """Error for an answer that hit the token limit: DegenerateOutputError when it ends in
    a whitespace run of at least `limit` characters (retried), else OutputTruncatedError
    (the content really did not fit; not retried)."""
    run = trailing_whitespace_run(text)
    if run >= limit:
        return DegenerateOutputError(
            f"{provider} answer degenerated into whitespace ({run} characters at the end, {reason})"
        )
    return OutputTruncatedError(f"{provider} answer cut off ({reason})")


class WhitespaceRunGuard:
    """Counts the whitespace run at the end of the text seen so far."""

    def __init__(self, limit: int = DEFAULT_MAX_WHITESPACE_RUN):
        self.limit = limit
        self.run = 0

    def feed(self, chunk: str) -> bool:
        """Adds a chunk; True when the run at the end reached the limit."""
        if not chunk:
            return False
        stripped = chunk.rstrip()
        if stripped:
            self.run = len(chunk) - len(stripped)
        else:
            self.run += len(chunk)
        return self.run >= self.limit
