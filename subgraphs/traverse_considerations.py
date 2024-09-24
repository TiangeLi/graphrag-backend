from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

from typing_extensions import TypedDict, Annotated
from operator import add

import json
from asyncio import gather

from helpers.constants import TRAV_TRAVERSALLLM, TRAV_PICKERLLM
from node_funcs.db_retriever import DBRetriever

retriever = DBRetriever()

@tool
async def get_considerations(target_node: str):
    """Use this tool to navigate a connected property graph of treatment considerations for BPH (benign prostate hyperplasia).
    This tool will help you find the relevant treatment considerations that are at play, to help guide the user on narrowing down their search on treatments for BPH.
    
    Here are the top level nodes that are available to you:
    <nodes>
    SURGICAL CONSIDERATIONS
    SYMPTOM PROFILE
    Presevation of Sexual Function (Ejaculatory & Erectile Function)
    MEDICAL COMPLEXITY / RISK
    PROSTATE SIZE / VOLUME
    </nodes>
    """
    parent = target_node
    result = await retriever.graph_traverse(
        start_node=target_node, 
        start_node_type='CONSIDERATION', 
        limit_rels='IS_TYPE_OF',
        depth=1)
    return parent, result

# ------------------------------------- #

traversal_template = \
"""You are a content expert on treatments for BPH (benign prostate hyperplasia).
Given the goal query, match the user's query to relevant treatment considerations.

You have access to a knowledge graph. Each node in this graph is a treatment consideration. 

The goal user query is:
<goal>
{goal}
</goal>

These nodes have already been selected for use:
<end_nodes>
{formatted_picked_nodes}
</end_nodes>

If the query is not fully covered by the above nodes, you can select additional nodes from the list below:
<nodes>
{formatted_path_nodes}
</nodes>

If you are finished finding considerations, return `done`"""

picker_template = \
"""Given the following user goal query:
<goal>
{goal}
</goal>

Here are some potentially relevant treatment considerations:
<considerations>
{formatted_end_nodes}
</considerations>

Please select the considerations that are most relevant to the user's query, which we will use to help guide the user to the best treatment. 
Stick to the provided list of considerations, do not make up any additional considerations not present and don't make conjectures about what may potentially be relevant. 

Only return the list of considerations. Do not return category headers or other text."""

# ------------------------------------- #

traversal_prompt = ChatPromptTemplate([
    ('placeholder', '{messages}'),
    ('system', traversal_template)
])

picker_prompt = ChatPromptTemplate([
    ('system', picker_template)
])

class PickedConsiderations(TypedDict):
    considerations: Annotated[list[str], ..., "The list of relevant considerations. Do not include categories."]

traversal_llm = TRAV_TRAVERSALLLM.bind_tools([get_considerations])
picker_llm = TRAV_PICKERLLM.with_structured_output(PickedConsiderations, method='json_schema', strict=True)

extractor_chain = traversal_prompt | traversal_llm
picker_chain = picker_prompt | picker_llm

# ------------------------------------- #

class SubState(TypedDict):
    raw_considerations: list[str]  # input from main graph
    collected_considerations: dict[str, list[str]]  # output to main graph
    # subgraph internal variables
    picked_considerations: Annotated[list[str], add]
    messages: Annotated[list[AnyMessage], add_messages]
    goal: str
    curr_end_nodes: list[str]
    curr_path_nodes: list[str]
    
async def format_raw_considerations_node(state: SubState):
    # TODO: we can do something more fancy than just joining. Maybe down the road.
    considerations = state.get("raw_considerations", [])
    numbered_considerations = "\n".join([f"{i+1}. {consideration}" for i, consideration in enumerate(considerations)])
    return {"messages": numbered_considerations, "goal": numbered_considerations}

async def traversal_node(state: SubState):
    formatted_picked_nodes = '\n'.join(state.get("picked_considerations", []))
    formatted_path_nodes = '\n'.join(state.get("curr_path_nodes", []))
    response = await extractor_chain.ainvoke({
        "messages": state["messages"],
        "goal": state["goal"],
        "formatted_picked_nodes": formatted_picked_nodes,
        "formatted_path_nodes": formatted_path_nodes
    })
    return {"messages": [response]}

async def use_tool_edge(state: SubState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "__traverse__"
    return "__done__"

async def tool_output_processor_node(state: SubState):
    tool_messages = []
    path_nodes = []
    end_nodes = []
    _other_end_nodes = []
    for message in reversed(state['messages']):
        if message.type == "tool":
            tool_messages.append(json.loads(message.content))
        else:
            break
    for category, considerations in tool_messages:
        if not considerations:
            _other_end_nodes.append(category)
        else:
            for parent, children in considerations.items():
                if not children:
                    _other_end_nodes.append(parent)
                else:
                    _end_nodes = []
                    for child in children:
                        if child['contains_considerations']:
                            path_nodes.append(child['node'])
                        if child['contains_treatment_recommendations']:
                            _end_nodes.append(child['node'])
                    if _end_nodes:
                        end_nodes.append(f"Category [{parent}]:")
                        end_nodes.extend(_end_nodes)
    if _other_end_nodes:
        end_nodes.append("Category [Other]:")
        end_nodes.extend(_other_end_nodes)
    return {"curr_end_nodes": end_nodes, "curr_path_nodes": path_nodes}

async def picker_node(state: SubState):
    if state["curr_end_nodes"]:
        response = await picker_chain.ainvoke({
            "goal": state["goal"],
            "formatted_end_nodes": "\n".join(state["curr_end_nodes"])
        })
        picked = [p for p in response['considerations'] if p not in state.get("picked_considerations", [])]
    else:
        picked = []
    return {"picked_considerations": picked}

async def exit_node(state: SubState):
    picked = state.get("picked_considerations", [])
    tasks = []
    for consideration in picked:
        tasks.append(retriever.graph_traverse(
            start_node=consideration,
            start_node_type='CONSIDERATION',
            limit_rels='RECOMMENDED_FOR',
            depth=1
        ))
    results = await gather(*tasks)
    for result in results:
        for consideration, treatments in result.items():
            result[consideration] = [t['node'] for t in treatments]
    return {"collected_considerations": results}

# ------------------------------------- #

builder = StateGraph(SubState)

builder.add_node("format_raw_considerations", format_raw_considerations_node)
builder.add_node("traversal_node", traversal_node)
builder.add_node("tool_get_considerations", ToolNode([get_considerations]))
builder.add_node("tool_output_processor", tool_output_processor_node)
builder.add_node("picker_node", picker_node)
builder.add_node("exit_node", exit_node)

builder.add_edge(START, "format_raw_considerations")
builder.add_edge("format_raw_considerations", "traversal_node")
builder.add_conditional_edges("traversal_node", use_tool_edge,
                              {"__traverse__": "tool_get_considerations",
                              "__done__": "exit_node"})
builder.add_edge("tool_get_considerations", "tool_output_processor")
builder.add_edge("tool_output_processor", "picker_node")
builder.add_edge("picker_node", "traversal_node")
builder.add_edge("exit_node", END)
considerations_graph = builder.compile()