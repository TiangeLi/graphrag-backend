from dotenv import load_dotenv
load_dotenv(override=True)

import streamlit as st
import json
from main_graph import graph
import asyncio

import os
import requests

backend_url = os.getenv("BACKEND_URL")

# ----------------------------------------

async def run_graph(input: dict):
    _ready_to_stream_response = False
    _plan_made = False
    discuss_treatments = False
    discuss_considerations = False
    treatments_run_id = None
    considerations_run_id = None
    _has_streamed_considerations = False
    async for event in graph.astream_events({"user_query": input}, version="v2"):
        if not _ready_to_stream_response:
            node = event.get("metadata", {}).get("langgraph_node", "")
            kind = event["event"]
            if node == "planner" and kind == "on_chat_model_end":
                _resp = json.loads(event["data"]["output"].content)
                discuss_treatments = _resp["specific_treatments"]
                discuss_considerations = _resp["treatment_considerations"]
                _plan_made = True
            elif node.startswith("response") and kind == "on_chat_model_start":
                if node.endswith("treatments"):
                    treatments_run_id = event["run_id"]
                elif node.endswith("considerations"):
                    considerations_run_id = event["run_id"]
            if _plan_made and all([
                (discuss_treatments and treatments_run_id) or not discuss_treatments,
                (discuss_considerations and considerations_run_id) or not discuss_considerations,
            ]):
                _ready_to_stream_response = True
        else:
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                node = event["metadata"]["langgraph_node"]
                if content and node.startswith("response"):
                    id = event["run_id"]
                    if not _has_streamed_considerations and discuss_considerations:
                        _has_streamed_considerations = True
                        yield json.dumps({
                            "type": "text",
                            "id": considerations_run_id,
                            "text": "---\n"
                        })
                    yield json.dumps({
                        "type": "text",
                        "id": id,
                        "text": content
                    })
            elif kind == "on_chat_model_end" and node == "suggest_qs":
                id = event["run_id"]
                _resp = json.loads(event["data"]["output"].content)
                for i, question in enumerate(_resp['questions']):
                    yield json.dumps({
                        "type": "suggestion",
                        "id": f'{id}_{i}',
                        "text": question
                    })
            elif kind == "on_tool_end":
                yield json.dumps({
                    "type": "tool_call_end",
                    "name": event["name"],
                    "args": event["data"].get("input"),
                    "result": event["data"].get("output").content,
                })



if "content" not in st.session_state:
    st.session_state.content = {}
if "containers" not in st.session_state:
    st.session_state.containers = {}
if "button_prompt" not in st.session_state:
    st.session_state.button_prompt = ""


def set_button_prompt(prompt):
    st.session_state.button_prompt = prompt

async def stream_results(input):
    with requests.post(backend_url, json={"user_query": input}, stream=True) as resp:
        for line in resp.iter_lines():
            st.write(line)
            result = json.loads(line)
            if not st.session_state.content.get(result['id']):
                st.session_state.content[result['id']] = {"text": "", "type": result['type']}
                st.session_state.containers[result['id']] = st.empty()
            st.session_state.content[result['id']]['text'] += result['text']
            
            if result['type'] == 'text':
                with st.session_state.containers[result['id']].chat_message('ai'):
                    st.markdown(st.session_state.content[result['id']]['text'])
                    "---"
            elif result['type'] == 'suggestion':
                st.button(result['text'], on_click=set_button_prompt, args=[result['text']], use_container_width=True)

st.set_page_config(layout="wide", page_title='Chat BPH Guidelines')
st.markdown('## Chat BPH Guidelines')
with st.chat_message('ai'):
    st.write('Hello! Ask me anything about Benign Prostatic Hyperplasia (BPH), including diagnosis, treatment options, and more.')

col1, col2 = st.columns(2)
with col1:
    left_container = st.empty()

if prompt := st.chat_input('Ask anything about BPH') or st.session_state.button_prompt:
    st.session_state.button_prompt = ""
    prompt = prompt or st.session_state.button_prompt
    with left_container.container():
        with st.chat_message('user'):
            st.markdown(prompt)
        with st.spinner('Generating...'):
            asyncio.run(stream_results(input=prompt)) 