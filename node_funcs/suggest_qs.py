from langchain_core.prompts import ChatPromptTemplate
from helpers.constants import SUGGESTERLLM

from typing_extensions import TypedDict, Annotated

class SuggestedQuestions(TypedDict):
    questions: list[str]

# ------------------------------------- #

suggestions_template = \
"""If you were in my situation, what else would you want to know?

Write 3 questions that you would want to ask if you were in my situation, in order of priority, to further understand the topic.
In general:
the first question should be about the most recommended treatment for my query / priorities,
the second question should be about another treatment in a similar fashion,
the third question should be about comparing and contrasting a topic group, such as treatments, side effects, or considerations, etc.

Keep the questions short and to the point."""

# ------------------------------------- #

suggestions_prompt = ChatPromptTemplate([
    ('human', '{query}'),
    ('ai', '{final_response}'),
    ("human", suggestions_template),
])

suggest_qs_chain = suggestions_prompt | SUGGESTERLLM.with_structured_output(SuggestedQuestions, method='json_schema', strict=True)