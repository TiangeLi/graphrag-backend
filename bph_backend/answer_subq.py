from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .constants import ANSWER_SUBQ_LLM

template = """The following is some context that was retrieved from a database.

Based on the context, please formulate a comprehensive answer to the question. Follow these instructions:
1. Before answering the question, carefully think about what will most comprehensively answer the question.
2. You can only use information from the provided <context>; do NOT use your prior knowledge or make up any information.
3. Use minimal markdown. 
4. Be concise and to the point, using short sentences.
5. You **MUST** answer comprehensively, so that the user is well informed of all relevant information!
6. Include all relevant data, statistics, and other objective information in your answer.
7. Do not make any extraneous commentary or assessments; only provide factual, objective information.

ABOVE ALL: BE ACCURATE AND AS COMPREHENSIVE AS POSSIBLE.

<context>
{context}
</context>"""

prompt = ChatPromptTemplate([
    #('human', '{query}'),
    #('ai', _prelim_search_result.content),
    ('system', template),
    ('human', '{query}'),
])

answer_subq_chain = prompt | ANSWER_SUBQ_LLM | StrOutputParser()







old_template = """The following is some context that was retrieved from a database.

Based on the context, please formulate a comprehensive answer to the question. Follow these instructions:
1. Before answering the question, carefully think about what will most comprehensively answer the question.
2. You can only use information from the provided <context>; do NOT use your prior knowledge or make up any information.
3. Use minimal markdown. 
4. Be concise and to the point, using short sentences.
5. You **MUST** answer comprehensively, so that the user is well informed of all relevant information!
6. Include all relevant data, statistics, and other objective information in your answer.
7. Do not make any extraneous commentary or assessments; only provide factual, objective information.

<context>
{context}
</context>"""
