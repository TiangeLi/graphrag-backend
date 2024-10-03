import uvicorn
import asyncio

from bph_backend.server import app as bph_app
from all_guidelines_backend.server import app as all_guidelines_app

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
