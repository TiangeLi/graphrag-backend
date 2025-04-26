from langchain_openai import ChatOpenAI

# ------------------------------------------------------------------- #
# OpenAI model parameters

LARGE_EMBD = 'text-embedding-3-large'
SMALL_EMBD = 'text-embedding-3-small'

BIG_MODEL = 'gpt-4.1'
SMALL_MODEL = 'gpt-4.1-nano'

# MAIN
CONVLLM = ChatOpenAI(model=BIG_MODEL, temperature=0.5, streaming=True)
CONVLLM_SMALL = ChatOpenAI(model=SMALL_MODEL, temperature=0.5, streaming=True)

# TRAVERSE CONSIDERATIONS
TRAV_TRAVERSALLLM = ChatOpenAI(model=SMALL_MODEL, temperature=0)
TRAV_PICKERLLM = ChatOpenAI(model=BIG_MODEL, temperature=0.3)

# PLANNER
PLANNERLLM = ChatOpenAI(model=BIG_MODEL, temperature=0.3)

# SUGGEST QUESTIONS
SUGGESTERLLM = ChatOpenAI(model=SMALL_MODEL, temperature=0.7)