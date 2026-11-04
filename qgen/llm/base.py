"""Interface of the LLM layer (Príloha B).

The pipeline talks only to these two protocols. Adapters in `qgen/llm/adapters/` implement
them for a vendor; tests use the replay client and fakes in `tests/`.

An adapter makes exactly one provider call per `generate` / `embed` call. Retries, waiting,
the concurrency limit and JSON parsing live in `qgen.llm.retry`, splitting texts into
embedding batches too. An adapter raises `qgen.llm.errors.ProviderError` (or a subclass)
for errors it can classify.
"""

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class LLMResponse:
    """`text` is the same field the pipeline used before (`response.text`).

    text   always a str; a blocked or empty answer is ""
    model  the model that answered
    usage  token counts when the provider returns them:
           input_tokens, output_tokens, total_tokens
    """

    text: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)


@runtime_checkable
class LLMClient(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        model: str,
        json_mode: bool = True,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse: ...


@runtime_checkable
class EmbeddingClient(Protocol):
    def embed(self, texts: list[str], *, model: str) -> list[list[float]]: ...
