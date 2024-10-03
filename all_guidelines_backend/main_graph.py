from dotenv import load_dotenv
load_dotenv('.env', override=True)
import os
from pathlib import Path

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY_ALL_GUIDELINES')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT_ALL_GUIDELINES')

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.messages import AnyMessage

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from operator import add
from typing_extensions import TypedDict, Annotated

from .constants import LARGE_EMBD
from .subquestions import subqueries_chain
from .answer_subq import answer_subq_chain
from .response import response_chain
from .query_from_history import query_from_history_chain
# ------------------------------------------------------------------- #

class MainState(TypedDict):
    messages: list[AnyMessage]

    subqueries: list[str]
    user_goal: str

    subresponses: Annotated[list[str], add]

    debug: Annotated[list, add]

class SubquestionState(TypedDict):
    user_goal: str
    subquery: str

# ------------------------------------------------------------------- #

class MainGraph(object):
    def __init__(self):
        current_dir = Path(__file__).resolve().parent
        vector_db = FAISS.load_local(current_dir / "docs_vector", 
                                       OpenAIEmbeddings(model=LARGE_EMBD), 
                                       allow_dangerous_deserialization=True)
        vector_retriever = vector_db.as_retriever(search_kwargs={"k": 50})
        self.reranker = CohereRerank(model="rerank-english-v3.0", top_n=5)
        self.retriever = ContextualCompressionRetriever(
            base_compressor=CohereRerank(model="rerank-english-v3.0", top_n=10), 
            base_retriever=vector_retriever
        )
        self.graph = self.build_graph()

    async def query_from_history(self, state: MainState):
        query = state["messages"][-1]['content']
        try:
            last_response = state["messages"][-2]['content']
        except IndexError:
            last_response = ""
        if not last_response:
            return
        query = await query_from_history_chain.ainvoke({"query": query, "last_response": last_response})
        return {"messages": [{"role": "human", "content": query}]}
    
    async def get_subquestions(self, state: MainState):
        return
        prompt = state["messages"][-1]['content']
        subqueries = await subqueries_chain.ainvoke({"query": prompt})
        return {"subqueries": subqueries["query_components"], "user_goal": subqueries["patient_query_goal"]}
    
    async def send_subquestions(self, state: MainState):
        return
        return [Send("answer_subquestion", {"subquery": s, "user_goal": state["user_goal"]}) for s in state["subqueries"]]
    
    async def answer_subquestion(self, state: SubquestionState):
        return
        query = f"{state['user_goal']} {state['subquery']}"
        refined_context = await self.retriever.ainvoke(query)
        #refined_context = await self.reranker.acompress_documents(context, query=state["subquery"])
        formatted_context = "\n\n".join([c.page_content for c in refined_context])
        subresponse = await answer_subq_chain.ainvoke({"query": query, "context": formatted_context})
        return {"subresponses": [subresponse], "debug": [len(refined_context)]}

    async def respond(self, state: MainState):
        prompt = state["messages"][-1]['content']
        refined_context = await self.retriever.ainvoke(prompt)
        formatted_context = "\n\n".join([c.page_content for c in refined_context])
        response = await response_chain.ainvoke({"query": prompt, "context": formatted_context})
        return {"messages": [response]}
        prompt = state["messages"][-1]['content']
        formatted_responses = "\n\n".join([f"<subquestion_{i+1}>:\n{subquery}\n{subresponse}</subquestion_{i+1}>" for i, (subquery, subresponse) in enumerate(zip(state["subqueries"], state["subresponses"]))])
        response = await response_chain.ainvoke({"query": prompt, "subresponses": formatted_responses})
        return {"messages": [response]}
    
    def build_graph(self):
        builder = StateGraph(MainState)
        builder.add_node('query_from_history', self.query_from_history)
        #builder.add_node('get_subquestions', self.get_subquestions)
        #builder.add_node('answer_subquestion', self.answer_subquestion)
        builder.add_node('respond', self.respond)

        builder.add_edge(START, 'query_from_history')
        builder.add_edge('query_from_history', 'respond')
        #builder.add_conditional_edges('get_subquestions', self.send_subquestions, ['answer_subquestion'])
        #builder.add_edge('answer_subquestion', 'respond')
        builder.add_edge('respond', END)
        graph = builder.compile()
        return graph
    
g = MainGraph()
graph = g.graph