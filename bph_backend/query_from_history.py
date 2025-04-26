from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .constants import QUERY_FROM_HISTORY_LLM

template = """Given the conversation history, rewrite the user's query so that it is not ambiguous and includes the most pertinent conversational context to inform how the user wants to be helped / what they want to know."""

prompt = ChatPromptTemplate.from_messages([
    ("ai", "{last_response}"),
    ("human", "{query}"),
    ("system", template),
])

query_from_history_chain = prompt | QUERY_FROM_HISTORY_LLM | StrOutputParser()
