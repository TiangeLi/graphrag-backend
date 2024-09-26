from langchain_core.prompts import ChatPromptTemplate
from helpers.constants import SUGGESTERLLM

from typing_extensions import TypedDict

class SuggestedQuestions(TypedDict):
    questions: list[str]

# ------------------------------------- #

suggestions_template = \
"""Write 3 follow-up questions the user can ask, based on the conversation so far.

The questions should be:
- Contextual to the user's overall goal and the information we have provided so far.
- Short and to the point.
- Not repetitive.
- Each question should be different. Focus each question on a specific treatment for BPH; pick the one that would be most relevant based on the conversation so far."""

# ------------------------------------- #

suggestions_prompt = ChatPromptTemplate([
    ('human', '{query}'),
    ('ai', '{final_response}'),
    ("human", suggestions_template),
])

suggest_qs_chain = suggestions_prompt | SUGGESTERLLM.with_structured_output(SuggestedQuestions, method='json_schema', strict=True)