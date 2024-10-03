from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import TypedDict, Optional, Annotated

from .constants import MAKE_SUBQ_LLM

class Plan(TypedDict):
    patient_query_goal: Annotated[Optional[str], ..., "The user's overall primary goal when asking this question. Specifically, what are they trying to achieve or find out?"]
    query_components: Annotated[Optional[list[str]], ..., "A list of subqueries extracted from the user query that we should use to answer this question."]


template = \
"""You are part of an expert knowledge agent, specializing in BPH (benign prostate hyperplasia).
Your task is specifically to create a plan for how best to answer a user's question.

Given the user query, decompose the query into a sub-questions that need to be answered. Specifically, we want to know about:

- patient_query_goal: what is the user trying to achieve with this query? be concise, in 5-10 words or less.
- query_components: what are the components of the user query? you can extract one or more components.

When extracting components, add each component to the patient_query_goal to create a standalone subquestion.

IMPORTANT: Do NOT get creative. Extract only from the provided query itself. Do not make up potential or possible parts of the query, just stick to what is provided."""

prompt = ChatPromptTemplate.from_messages([
    ('system', template),
    ('human', '{query}')
])

subqueries_chain = prompt | MAKE_SUBQ_LLM.with_structured_output(Plan, method='json_schema', strict=True)
