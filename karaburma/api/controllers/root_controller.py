from fastapi import APIRouter, status
from fastapi.responses import JSONResponse


root_router = APIRouter(prefix="/api/v1", tags=["root"])

@root_router.get("/", status_code=status.HTTP_200_OK)
async def root_selfcheck():
    return JSONResponse(content={"message": "Uvicorn server was started for Karaburma."})