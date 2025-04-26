import uvicorn
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from bph_backend.server import run_graph as run_graph_bph
from all_guidelines_backend.server import run_graph as run_graph_all_guidelines

origins = [
    "http://localhost:8501", 
    "http://localhost:3000",
    "https://tli.koyeb.app",
]

bph_app = FastAPI()
all_guidelines_app = FastAPI()

bph_app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

all_guidelines_app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

"""@bph_app.post("/chat")
async def chat(input: dict):
    print("BPH: ", input)
    try:
        input = input.get("messages", [])[-1]['content'][0]['text']
    except IndexError as e:
        input = input
    return StreamingResponse(run_graph_bph(input))"""


@bph_app.post("/chat")
async def chat(input: dict):
    print("BPH: ", input)
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
    return StreamingResponse(run_graph_bph(payload))

@all_guidelines_app.post("/chat")
async def chat(input: dict):
    print("CUA: ", input)
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
    return StreamingResponse(run_graph_all_guidelines(payload))


async def run_servers():
    config_bph = uvicorn.Config(bph_app, host="0.0.0.0", port=8000)
    config_all_guidelines = uvicorn.Config(all_guidelines_app, host="0.0.0.0", port=8001)

    server_bph = uvicorn.Server(config_bph)
    server_all_guidelines = uvicorn.Server(config_all_guidelines)

    await asyncio.gather(
        server_bph.serve(),
        server_all_guidelines.serve(),
    )

if __name__ == "__main__":
    asyncio.run(run_servers())
