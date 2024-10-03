from dotenv import load_dotenv
import os
load_dotenv('.env', override=True)

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY_BPH')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT_BPH')

from asyncio import gather

from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from .subgraphs.traverse_considerations import considerations_graph
from .node_funcs.db_retriever import DBRetriever
from .node_funcs.planner import planner_chain
from .node_funcs.form_response import considerations_chain, treatments_chain
from .node_funcs.suggest_qs import suggest_qs_chain

from .helpers.utils import a_cluster_strings
from .helpers.constants import CONVLLM_SMALL
# ------------------------------------------------------------------- #

class MainState(TypedDict):
    user_query: str

    specific_treatments: list[str]
    raw_considerations: list[str]
    collected_considerations: list[str]

    formatted_treatments: list
    formatted_considerations: dict

    considerations_response: str
    treatments_response: str
    final_response: str
    suggested_qs: list[str]

    scratchpad: str

class MainGraph(object):
    def __init__(self):
        self.db_retriever = DBRetriever()
        self.graph = self.build_graph()
    
    async def planning_node(self, state: MainState):
        response = await planner_chain.ainvoke({"user_query": state["user_query"]})
        specific_treatments = response.get("specific_treatments", [])
        if specific_treatments:
            specific_treatments = [tx.upper() for tx in specific_treatments]
        raw_considerations = response.get("treatment_considerations", [])
        return {"specific_treatments": specific_treatments, "raw_considerations": raw_considerations, "scratchpad": response}
    
    async def planner_edge(self, state: MainState):
        paths = []
        if state['specific_treatments']:
            paths.append("_treatments_")
        if state['raw_considerations']:
            paths.append("_considerations_")
        if not paths:
            paths.append("__done__")
        return paths
    
    async def get_specific_treatments(self, state: MainState):
        if state.get("specific_treatments"):
            response = await gather(*[self.db_retriever.a_get_tx_name_by_similarity(tx, top_k=2) for tx in state["specific_treatments"]])
            return {"specific_treatments": response}
        else:
            return {"specific_treatments": []}
        
    async def format_treatments(self, state: MainState):
        treatments = state.get("specific_treatments", [])
        formatted_treatments = []
        if treatments:
            for treatment in treatments:
                for _treatment_name, _guidelines in treatment.items():
                    guidelines = []
                    for i, guideline in enumerate(_guidelines):
                        formatted_guideline = f"#{i+1}: {guideline['name']}\n{guideline['content']}\n(metadata: {guideline['metadata']})"
                        guidelines.append(formatted_guideline)
                    formatted_treatments.append(f"Guideline Exerpts for {_treatment_name}:\n\n{'\n\n'.join(guidelines)}")
        formatted_treatments = '\n\n---\n\n'.join(formatted_treatments)
        return {"formatted_treatments": formatted_treatments}
        
    async def format_considerations(self, state: MainState):
        def _iter_tree(treatments, formatted_container):
            for treatment in treatments:
                if isinstance(treatment, dict):
                    if treatment['children']:
                        container = []
                        container = _iter_tree(treatment['children'], container)
                        formatted_container.append({treatment['name']: container})
                    else:
                        formatted_container.append(treatment['name'])
                else:
                    formatted_container.append(treatment)
            return formatted_container

        considerations = state.get("collected_considerations", [])
        formatted_considerations = {}
        if considerations:
            for consideration in considerations:
                for name, treatment_names in consideration.items():
                    sorted_treatments = await self.db_retriever.a_sort_tx_by_type(treatment_names)
                    formatted_considerations[name] = {}
                    for category, treatments in sorted_treatments.items():
                        formatted_considerations[name][category] = _iter_tree(treatments, [])

        if not formatted_considerations:
            return {"formatted_considerations": ""}

        clustered = await a_cluster_strings([name for name in formatted_considerations.keys()])
        clustered_formatted_considerations = {}
        for consideration in clustered:
            clustered_formatted_considerations[consideration] = formatted_considerations[consideration]

        formatted_string = ""
        for i, (consideration, treatments) in enumerate(clustered_formatted_considerations.items()):
            formatted_string += f"Consideration #{i+1}: {consideration}\n"
            formatted_string += "```(treatment_recommendations)\n"
            for category, treatments in treatments.items():
                formatted_string += f"{category} Treatments {{{{\n"
                for treatment in treatments:
                    formatted_string += f"- {treatment}\n"
                formatted_string += "}}\n"
            formatted_string += "```\n"

        return { "formatted_considerations": formatted_string}
        
    async def response_treatments_node(self, state: MainState):
        specific_treatments = [tx_name for treatment in state.get("specific_treatments", []) for tx_name in treatment.keys()]
        formatted_treatments = state.get("formatted_treatments", [])
        query = state.get("user_query", "")
        response = ""
        if formatted_treatments:
            response = await treatments_chain.ainvoke({
                "query": query,
                "specific_treatments": specific_treatments,
                "formatted_treatments": formatted_treatments,
            })
            response = response.content
        return {"treatments_response": response}
    
    async def response_considerations_node(self, state: MainState):
        formatted_considerations = state.get("formatted_considerations", {})
        specific_treatments = [tx_name for treatment in state.get("specific_treatments", []) for tx_name in treatment.keys()]
        query = state.get("user_query", "")
        response = ""
        if formatted_considerations:
            response = await considerations_chain.ainvoke({
                "query": query,
                "formatted_considerations": formatted_considerations,
                "specific_treatments": specific_treatments,
            })
            response = response.content
        return {"considerations_response": response}
    
    async def response_collector_node(self, state: MainState):
        query = state.get("user_query", "")
        responses = [state.get("considerations_response", ""), state.get("treatments_response", "")]
        responses = [response for response in responses if response]
        if responses:
            final_response = ""
            for i, response in enumerate(responses):
                if i > 0:
                    final_response += "\n\n---\n\n"
                final_response += f"{response}"
        else:
            redirect_template = "You are an assistant that helps patients find information about treatments for BPH.\n"
            redirect_template += "The user's query was not relevant to BPH. Please politely notify the user of this but do not specifically answer their irrelevant query.\n"
            redirect_template += "Instead, kindly redirect them by mentioning the main purpose of this chatbot - helping patients find information about treatments for BPH.\n"
            redirect_template += "You will be provided with the user's query, and you should use it to guide the redirection.\n\n"
            redirect_template += f"Here is the user's query:\n\nUser: {query}"
            final_response = await CONVLLM_SMALL.ainvoke(redirect_template)
        return {"final_response": final_response}
    
    async def suggest_qs_node(self, state: MainState):
        query = state.get("user_query", "")
        final_response = state.get("final_response", "")
        response = await suggest_qs_chain.ainvoke({"query": query, "final_response": final_response})
        return {"suggested_qs": response['questions']}
    
    def build_graph(self):
        builder = StateGraph(MainState)
        builder.add_node("planner", self.planning_node)
        builder.add_node("get_specific_treatments", self.get_specific_treatments)
        builder.add_node("get_treatment_considerations", considerations_graph)
        builder.add_node("format_treatments", self.format_treatments)
        builder.add_node("format_considerations", self.format_considerations)
        builder.add_node("response_treatments", self.response_treatments_node)
        builder.add_node("response_considerations", self.response_considerations_node)
        builder.add_node("response_collector", self.response_collector_node)
        builder.add_node("suggest_qs", self.suggest_qs_node)
        
        builder.add_edge(START, "planner")
        builder.add_conditional_edges("planner", self.planner_edge,
                                      {"_treatments_": "get_specific_treatments",
                                       "_considerations_": "get_treatment_considerations",
                                       "__done__": "response_collector"})
        builder.add_edge("get_specific_treatments", "format_treatments")
        builder.add_edge("get_treatment_considerations", "format_considerations")
        builder.add_edge("format_treatments", "response_treatments")
        builder.add_edge("format_considerations", "response_considerations")
        builder.add_edge("response_treatments", "response_collector")
        builder.add_edge("response_considerations", "response_collector")
        builder.add_edge("response_collector", "suggest_qs")
        builder.add_edge("suggest_qs", END)

        graph = builder.compile()
        return graph
    
g = MainGraph()
graph = g.graph