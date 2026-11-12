"""Pipeline results for the output tests (qgen.output): sections, questions, settings."""

from qgen.config import Settings
from qgen.lo.pipeline import LOPipelineResult
from qgen.output import build_output
from qgen.pipeline import PipelineResult
from tests.fakes import FakeEmbedder, FakeLLM, TitlesLLM


def settings(**values) -> Settings:
    return Settings(_env_file=None, **values)


def section(lo_id, name, bloom="Pochopiť", prerequisites=()):
    return {"id": lo_id, "vzdelávací_objekt": name, "bloom_level": bloom, "odporúčané_zadania": "Vysvetli.",
            "citovane_zdroje": [f"D1:{lo_id}"], "prerekvizity": list(prerequisites)}


def question(lo_id, description="Čo je to?", answer="Toto.", hints=("<p>eko.pdf, strana 1</p>",), zdroje=("D1:1",),
             question_type="open", name="", bloom="understand", keywords=("pojem",), choices=None):
    """A question in the internal format of the pipeline (qgen.questions.types)."""
    if choices is None:
        choices = [{"text": answer, "is_correct": True}] if question_type == "open" else []
    return {"id": lo_id, "lo_id": lo_id, "name": name, "description": description, "type": question_type,
            "time_to_handle": 300, "time_to_explain": 60,
            "hints": [{"name": "Zdroj", "url": h} for h in hints], "choices": choices,
            "bloom_level": bloom, "keywords": list(keywords), "source": list(zdroje), "practical": False}


def result(sections, questions, reports=None, language="sk") -> PipelineResult:
    lo_result = LOPipelineResult.__new__(LOPipelineResult)
    lo_result.language_info = {"language": language, "reason": "test"}
    return PipelineResult(learning_objects=sections, questions=questions, reports=reports or {}, lo_result=lo_result)


def build(sections, questions, *, names=("a.pdf",), reports=None, language="sk", llm=None, timings=None,
          titles=True, embeddings=None, **kwargs):
    """build_output on a fake pipeline result. `llm` gets every call except the lesson
    titles, which TitlesLLM answers (titles=False: `llm` gets those too)."""
    llm = llm if llm is not None else FakeLLM()
    pipeline_result = result(sections, questions, reports, language)
    pipeline_result.lo_result.embeddings = embeddings
    return build_output(
        pipeline_result,
        input_names=list(names),
        timings=timings if timings is not None else {},
        settings=kwargs.pop("settings", None) or settings(),
        llm=TitlesLLM(llm) if titles else llm,
        embedder=kwargs.pop("embedder", None) or FakeEmbedder(),
        **kwargs,
    )
