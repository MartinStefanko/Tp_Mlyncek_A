"""Small builders for the split tests (times in minutes, pages in D1 unless given)."""

from qgen.contract.options import LessonsSplit, SingleSplit, SyllabusSplit, TopicsSplit
from qgen.split import SplitSection


def section(sid, minutes, prerequisites=(), page=None, topic=None, pages=None):
    if pages is None:
        pages = (("D1", page if page is not None else sid),)
    return SplitSection(
        id=sid,
        prerequisites=tuple(prerequisites),
        pages=tuple(pages),
        seconds=round(minutes * 60),
        topic=topic,
    )


def chain(minutes, topics=None):
    """Sections 1..n in a prerequisite chain, page = id."""
    return [
        section(i, m, prerequisites=(i - 1,) if i > 1 else (), topic=topics[i - 1] if topics else None)
        for i, m in enumerate(minutes, start=1)
    ]


def single(minutes=45):
    return SingleSplit(mode="single", lesson_minutes=minutes)


def lessons(count=None, minutes=45):
    return LessonsSplit(mode="lessons", lesson_count=count, lesson_minutes=minutes)


def topics(minutes=45):
    return TopicsSplit(mode="topics", lesson_minutes=minutes)


def syllabus(*items, minutes=45):
    return SyllabusSplit(mode="syllabus", lesson_minutes=minutes, syllabus=list(items))


def syllabus_text(text="3 hodiny: najprv teória, potom príklady", minutes=45):
    return SyllabusSplit(mode="syllabus", lesson_minutes=minutes, syllabus_text=text)


def ids(result):
    return [list(lesson.section_ids) for lesson in result.lessons]


def codes(result):
    return [w.code for w in result.warnings]
