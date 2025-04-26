from dotenv import load_dotenv
load_dotenv(override=True)

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from .main_graph import graph
import json
# ----------------------------------------

async def run_graph(input):
    async for event in graph.astream_events({"messages": input}, version="v2"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            node = event["metadata"]["langgraph_node"]
            if content and node == "respond":
                id = event["run_id"]
                yield json.dumps({"id": id, "text": content, "type": "text"})