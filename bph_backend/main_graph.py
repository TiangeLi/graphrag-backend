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
from .choose_tx import choose_tx_chain
# ------------------------------------------------------------------- #

class MainState(TypedDict):
    messages: list[AnyMessage]

    treatments_to_discuss: list[str]

    subqueries: list[str]
    user_goal: str

    subresponses: Annotated[list[str], add]

    debug: Annotated[list, add]

class SubquestionState(TypedDict):
    user_goal: str
    subquery: str

# ------------------------------------------------------------------- #

import pickle
from langchain.storage import InMemoryByteStore
from langchain_community.vectorstores import FAISS  
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.retrievers.multi_vector import SearchType
from langchain_openai import OpenAIEmbeddings
pickle_directory = "aua"

class MainGraph(object):
    def __init__(self):
        current_dir = Path(__file__).resolve().parent
        #vector_db = FAISS.load_local(current_dir / "docs_vector", 
        #                               OpenAIEmbeddings(model=LARGE_EMBD), 
        #                               allow_dangerous_deserialization=True)
        #vector_retriever = vector_db.as_retriever(search_kwargs={"k": 50})


        with open(f'{current_dir}/{pickle_directory}/doc_ids.pkl', 'rb') as file:
            doc_ids = pickle.load(file)
        with open(f'{current_dir}/{pickle_directory}/summary_docs.pkl', 'rb') as file:
            summary_docs = pickle.load(file)
        with open(f'{current_dir}/{pickle_directory}/docs.pkl', 'rb') as file:
            docs = pickle.load(file)

        vectorstore = FAISS.from_documents(summary_docs, OpenAIEmbeddings(model=LARGE_EMBD))
        store = InMemoryByteStore()
        id_key = 'doc_id'
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            byte_store=store,
            id_key=id_key,
            search_type=SearchType.similarity,
            search_kwargs={'k': 20}
        )
        retriever.docstore.mset(list(zip(doc_ids, docs)))

        self.big_retriever = ContextualCompressionRetriever(
            base_compressor=CohereRerank(model="rerank-v3.5", top_n=10), 
            base_retriever=retriever
        )

        self.small_retriever = ContextualCompressionRetriever(
            base_compressor=CohereRerank(model="rerank-v3.5", top_n=4), 
            base_retriever=retriever
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
    
    async def get_treatments_to_discuss(self, state: MainState):
        prompt = state["messages"][-1]['content']
        treatments_to_discuss = await choose_tx_chain.ainvoke({"query": prompt})
        treatments_to_discuss = treatments_to_discuss["treatments_to_discuss"]
        return {"treatments_to_discuss": treatments_to_discuss}

    async def respond(self, state: MainState):
        prompt = state["messages"][-1]['content']
        treatments_to_discuss = state["treatments_to_discuss"]

        if treatments_to_discuss == []:
            refined_context = await self.big_retriever.ainvoke(prompt)
            context_hint = ""
        else:
            refined_context = await self.big_retriever.ainvoke(prompt)
            extra_context = await self.small_retriever.abatch(treatments_to_discuss)
            flattened_context = []
            for context_list in [refined_context]+extra_context:
                flattened_context.extend(context_list)
            refined_context = flattened_context
            context_hint = f"You should include a discussion of the following in your response, given their particular relevance to the user's current query: {', '.join(treatments_to_discuss)}"

        formatted_context = "\n\n".join([c.page_content for c in refined_context])
        response = await response_chain.ainvoke({"query": prompt, "context": formatted_context, "context_hint": context_hint})
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
        builder.add_node('get_treatments_to_discuss', self.get_treatments_to_discuss)
        builder.add_node('respond', self.respond)

        builder.add_edge(START, 'query_from_history')
        builder.add_edge('query_from_history', 'get_treatments_to_discuss')
        builder.add_edge('get_treatments_to_discuss', 'respond')
        #builder.add_conditional_edges('get_subquestions', self.send_subquestions, ['answer_subquestion'])
        #builder.add_edge('answer_subquestion', 'respond')
        builder.add_edge('respond', END)
        graph = builder.compile()
        return graph
    
g = MainGraph()
graph = g.graph