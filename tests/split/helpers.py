"""Small builders for the split tests (times in minutes, pages in D1 unless given)."""

from qgen.contract.options import LessonsSplit, SingleSplit, SyllabusSplit, TopicsSplit
from qgen.split import SplitSection


def section(sid, minutes, prerequisites=(), page=None, pages=None):
    if pages is None:
        pages = (("D1", page if page is not None else sid),)
    return SplitSection(id=sid, prerequisites=tuple(prerequisites), pages=tuple(pages), seconds=round(minutes * 60))


def chain(minutes):
    """Sections 1..n in a prerequisite chain, page = id."""
    return [section(i, m, prerequisites=(i - 1,) if i > 1 else ()) for i, m in enumerate(minutes, start=1)]


def topical(labels, ids=None):
    """Keyword arguments for split_lessons: topic names and a one-hot similarity per section
    (label None = the section has no similarity row)."""
    names = sorted({label for label in labels if label is not None})
    ids = ids or range(1, len(labels) + 1)
    similarity = {
        sid: [1.0 if name == label else 0.0 for name in names] for sid, label in zip(ids, labels) if label is not None
    }
    return {"topics": names, "topic_similarity": similarity}


def single():
    return SingleSplit(mode="single")


def lessons(count):
    return LessonsSplit(mode="lessons", lesson_count=count)


def topics():
    return TopicsSplit(mode="topics")


def syllabus(*items):
    return SyllabusSplit(mode="syllabus", syllabus=list(items))


def syllabus_text(text="3 hodiny: najprv teória, potom príklady"):
    return SyllabusSplit(mode="syllabus", syllabus_text=text)


def ids(result):
    return [list(lesson.section_ids) for lesson in result.lessons]


def codes(result):
    return [w.code for w in result.warnings]
