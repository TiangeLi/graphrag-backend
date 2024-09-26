from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import TypedDict, Optional, Annotated, Literal

from helpers.constants import PLANNERLLM

class Plan(TypedDict):
    patient_query_goal: Annotated[Optional[Literal["topic_overview", "understand_treatment_considerations", "understand_specific_treatments"]], ..., "The user's goal in 2 words or less."]
    treatment_considerations: Annotated[Optional[list[str]], ..., "A list of considerations extracted from the user query that we should use to narrow down specific treatment recommendations."]
    asked_about_specific_treatments: Annotated[bool, ..., "True if the user query is about specific treatments by treatment name, False otherwise."]
    specific_treatments: Annotated[Optional[list[str]], ..., "A list of BPH treatments that the user specifically asked about by name."]


template = \
"""You are part of an expert knowledge agent, specializing in BPH (benign prostate hyperplasia).
Your task is specifically to create a plan for how best to answer a user's question.

Given the user query, decompose the query into a sub-questions that need to be answered. Specifically, we want to know about:

- patient_query_goal: what is the user trying to achieve with this query? use 5 words or less.
- treatment_considerations: if the user question is more open-ended, then we don't know what specific treatments to discuss yet. In this case, we will engage with treatment considerations in order to narrow down the best treatments to discuss.
- asked_about_specific_treatments: a boolean flag. Did the user ask about any specific treatments, specifically by name? Return False if you cannot recognize any named treatment entities in the user query.
- specific_treatments: an optional field. If asked_about_specific_treatments is True, then return a list of BPH treatments that the user specifically asked about by name. If asked_about_specific_treatments is False, then return null

IMPORTANT: FOR TREATMENT CONSIDERATIONS: Do NOT get creative. Extract considerations only from the provided query itself. Do not make up potential or possible considerations, just stick to what is provided."""

prompt = ChatPromptTemplate.from_messages([
    ('system', template),
    ('human', '{user_query}')
])

planner_chain = prompt | PLANNERLLM.with_structured_output(Plan, method='json_schema', strict=True)