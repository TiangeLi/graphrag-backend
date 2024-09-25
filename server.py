from dotenv import load_dotenv
load_dotenv(override=True)
from os import getenv

from langchain_core.messages import HumanMessage, AIMessage

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
from main_graph import graph
# ----------------------------------------

origins = [getenv("FRONTEND_URL"), "http://localhost:8501", "https://bph.koyeb.app"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    
)

def convert_to_langchain_messages(messages):
    langchain_messages = []
    for message in messages:
        if message.type == "human":
            langchain_messages.append(HumanMessage(content=message.content[0]['text'], id=message.id))
        elif message.type == "ai":
            content = ''
            for item in message.content:
                if item['type'] == 'text' and item.get('text'):
                    content += item['text'] + '\n\n'
            langchain_messages.append(AIMessage(content=content, id=message.id))
    return langchain_messages


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
                            "text": ""
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

@app.post("/chat")
async def chat(input: dict):
    try:
        input = input.get("messages", [])[-1]['content'][0]['text']
    except IndexError as e:
        input = input
    return StreamingResponse(run_graph(input))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)