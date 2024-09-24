from langchain_core.prompts import ChatPromptTemplate
from helpers.constants import EXTRACTORLLM
from node_funcs.db_retriever import DBRetriever
from langchain_core.tools import tool

retriever = DBRetriever()

@tool
async def get_considerations(target_node: str):
    """Use this tool when the user query is ambiguous, 
    so that we can learn about the treatment considerations that are in play to help guide the user.
    
    Here are the top level nodes we can traverse to:
    <nodes>
    TREATMENT CONSIDERATIONS
    - SURGICAL CONSIDERATIONS
    - SYMPTOM PROFILE
    - Presevation of Sexual Function (Ejaculatory & Erectile Function)
    - MEDICAL COMPLEXITY / RISK
    - PROSTATE SIZE / VOLUME    
    </nodes>"""
    result = await retriever.graph_traverse(start_node=target_node, start_node_type='CONSIDERATION', depth=2)
    return result

@tool
async def learn_about_treatments(target_node: str):
    """Use this tool if the user query is about specific treatments,
    or when you have a list of treatments (e.g. from get_considerations) and want to learn more about them.
    
    You can only use this tool if the target node has label "SURGICAL", "PHARMACOLOGICAL", or "CONSERVATIVE".

    Here are the top level nodes we can traverse to:
    <nodes>
    SURGICAL TREATMENT
    - ENUCLEATION OF THE PROSTATE
    - RESECTION OF THE PROSTATE
    - VAPORIZATION OF THE PROSTATE
    - ALTERNATIVE ABLATIVE TECHNIQUES
    - NON-ABLATIVE TECHNIQUES
    PHARMACOLOGICAL TREATMENT
    - MONOTHERAPY
    - COMBINATION THERAPY
    CONSERVATIVE TREATMENT
    - Watchful Waiting
    - Behavioural and Dietary Modifications
    </nodes>"""
    result = await retriever.graph_traverse(start_node=target_node, start_node_type='TREATMENT', depth=2)
    return result

template = """You are a content expert on treatments for BPH (benign prostate hyperplasia).
You have access to a knowledge graph with the following top level categories to start with:

Given the user query, select the relevant categories we should traverse to.

Remember: you do not have to use all of the information returned if not relevant. 
You can go as deep as necessary to find the information you need.
You can backtack and go to previous nodes if needed to find new paths.

Focus on answering the question rather than summarizing returned nodes.

You should use get_considerations to find a list of treatments based on relevant considerations,
then (the sequence is important here) you should use learn_about_treatments to get more information about the treatments."""

prompt = ChatPromptTemplate.from_messages([
    ('system', template),
    ('placeholder', '{messages}')
])

extractor_llm = EXTRACTORLLM.bind_tools([get_considerations, learn_about_treatments])

extractor_chain = prompt | extractor_llm