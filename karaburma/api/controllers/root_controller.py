import asyncio
import os
import signal

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse


root_router = APIRouter(prefix="/api/v1", tags=["root"])

@root_router.get("/", status_code=status.HTTP_200_OK)
async def root_selfcheck():
    return JSONResponse(content={"message": "Uvicorn server was started for Karaburma."})

@root_router.get("/shutdown", status_code=status.HTTP_200_OK)
async def root_shutdown():
    response_content = {"message": "Uvicorn server is going to stop. Please wait a few seconds."}
    await asyncio.sleep(0.5)
    os.kill(os.getpid(), signal.SIGTERM)
    return JSONResponse(content=response_content)