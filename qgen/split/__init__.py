"""Split into lessons (kap. 8): pure functions, no LLM and no embeddings.

    build_split_sections   pipeline sections + questions → SplitSection (time, pages, prerequisites)
    split_lessons          sections + options.split (+ similarity, syllabus plan) → lessons, warnings
    order_questions        order of the questions in each lesson (choice → open → no_answer)
    SyllabusPlan           structured syllabus (from split.syllabus or, in step 16, from syllabus_text)

Step 16 connects it to the output (topics from embeddings, the similarity matrix, the plan
from the syllabus text, title and description of a lesson by the LLM).
"""

from qgen.split.lessons import Lesson, SplitResult, order_questions, split_lessons
from qgen.split.order import SectionOrder, order_sections
from qgen.split.sections import SplitQuestion, SplitSection, build_split_sections, page_sort_key
from qgen.split.syllabus import PlannedLesson, SyllabusPlan, plan_from_syllabus

__all__ = [
    "Lesson",
    "PlannedLesson",
    "SectionOrder",
    "SplitQuestion",
    "SplitResult",
    "SplitSection",
    "SyllabusPlan",
    "build_split_sections",
    "order_questions",
    "order_sections",
    "page_sort_key",
    "plan_from_syllabus",
    "split_lessons",
]
