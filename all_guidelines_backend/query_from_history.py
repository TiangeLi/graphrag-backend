from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .constants import QUERY_FROM_HISTORY_LLM

template = """Given the conversation history, rewrite the user's query in a way that is concise and captures the essence of their question, in the context of the conversation history."""

prompt = ChatPromptTemplate.from_messages([
    ("ai", "{last_response}"),
    ("human", "{query}"),
    ("system", template),
])

query_from_history_chain = prompt | QUERY_FROM_HISTORY_LLM | StrOutputParser()
