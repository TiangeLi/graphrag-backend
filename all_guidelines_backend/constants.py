from langchain_openai import ChatOpenAI


LARGE_EMBD = "text-embedding-3-large"

__BIG_MODEL = "gpt-4.1"
__SMALL_MODEL = "gpt-4o-mini"

#__EXPERIMENTAL_MODEL = "chatgpt-4o-latest"

# ------------------------------------------------------------------- #

MAKE_SUBQ_LLM = ChatOpenAI(model=__BIG_MODEL, temperature=0.5)
ANSWER_SUBQ_LLM = ChatOpenAI(model=__SMALL_MODEL, temperature=0.5)

CONV_LLM = ChatOpenAI(model=__BIG_MODEL, temperature=0.2)

QUERY_FROM_HISTORY_LLM = ChatOpenAI(model=__SMALL_MODEL, temperature=0.3)