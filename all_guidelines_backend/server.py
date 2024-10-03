from dotenv import load_dotenv
load_dotenv(override=True)

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from .main_graph import graph
import json
# ----------------------------------------

origins = [
    "http://localhost:8501", 
    "http://localhost:3000",
    "https://tli.koyeb.app"
    "https://tli.koyeb.app/chat?server=bph",
    "https://tli.koyeb.app/chat?server=all_guidelines",
    "https://tli.koyeb.app/chat",
    ]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    
)
async def run_graph(input):
    async for event in graph.astream_events({"messages": input}, version="v2"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            node = event["metadata"]["langgraph_node"]
            if content and node == "respond":
                id = event["run_id"]
                yield json.dumps({"id": id, "text": content, "type": "text"})

@app.post("/chat")
async def chat(input: dict):
    last_response = ""
    try:
        input = input.get("messages", [])
        query = input[-1]['content'][0]['text']
        if len(input) > 1:
            last_response = input[-2]['content'][0]['text']
    except IndexError as e:
        query = input
    payload = [
        {"role": "ai", "content": last_response},
        {"role": "human", "content": query}
    ]
    return StreamingResponse(run_graph(payload))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)