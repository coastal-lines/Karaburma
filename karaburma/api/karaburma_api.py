import argparse
import asyncio
import os
import traceback
import uvicorn
from http.client import HTTPException
from fastapi import FastAPI, Request, status, APIRouter, Depends
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse

from karaburma.api.controllers.base64_controller import Base64Controller
from karaburma.api.controllers.file_controller import FileController
from karaburma.api.services.base64_service import Base64Service
from karaburma.api.services.file_service import FileService
from karaburma.api.controllers.root_controller import root_router
from karaburma.api.controllers.screenshot_controller import ScreenshotController
from karaburma.api.services.screenshot_service import ScreenshotService
from karaburma.utils import files_helper
from karaburma.main import Karaburma


class KaraburmaApiService:
    def __init__(self, host, port: int, config_path: str, source_mode: str, detection_mode: str, logging: bool):
        self.__server = None
        self.__server_task = None

        self.__host = host
        self.__port = port

        self.__karaburma_instance = Karaburma(config_path, source_mode, detection_mode, logging)

        self.__app = FastAPI()

        @self.__app.exception_handler(400)
        async def not_found_exception_handler(request: Request, exc: HTTPException):
            return JSONResponse(status_code=400, content={"message": f"Please check ."})

        @self.__app.exception_handler(404)
        async def not_found_exception_handler(request: Request, exc: HTTPException):
            return JSONResponse(status_code=404, content={"message": f"Url '{str(request.url)}' not found."})

        @self.__app.exception_handler(405)
        async def not_found_exception_handler(request: Request, exc: HTTPException):
            return JSONResponse(status_code=405,
                                content={"message": f"Url '{str(request.url)}' not found."})

        # Handler for RequestValidationError
        # FastApi has built-in handle for validating Pydantic model.
        # Pydantic is a data validation and settings management library in Python.
        @self.__app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, exc: RequestValidationError):
            return JSONResponse(status_code=422,
                                content={"message": "Please check json values for your POST request.",
                                         "details": exc.errors()})

        # Handler for server exceptions
        @self.__app.exception_handler(Exception)
        async def generic_exception_handler(request: Request, exc: Exception):
            traceback_str = "".join(traceback.format_tb(exc.__traceback__))
            return JSONResponse(
                status_code=500,
                content={"message": "Internal server error",
                         "error": str(exc),
                         "traceback": f"{exc}\n\n{traceback_str}"})

    @property
    def host(self) -> str:
        return self.__host

    @host.setter
    def host(self, host: str):
        self.__host = host

    @property
    def port(self) -> int:
        return self.__port

    @port.setter
    def port(self, port: int):
        self.__port = port

    def get_app(self):
        return self.__app

    # For starting server by command line
    async def start_karaburma_service(self):
        uvicorn_config = uvicorn.Config(app=self.__app, host=self.__host, port=self.__port)
        self.__server = uvicorn.Server(uvicorn_config)
        await self.__server.serve()

    # For starting server by test fixtures
    def start_karaburma_service_for_test_fixtures(self):
        uvicorn_config = uvicorn.Config(app=self.__app, host=self.__host, port=self.__port, log_level="trace")
        self.__server = uvicorn.Server(uvicorn_config)
        task = asyncio.create_task(self.__server.serve())

        return task

    async def stop_karaburma_service(self):
        if self.__server is not None:
            self.__server.should_exit = True

            # Wait few seconds for stop
            await asyncio.sleep(2)

            self.__server = None

if __name__ == "__main__":
    # Create command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', help='host')
    parser.add_argument('--port', help='port')
    parser.add_argument('--source_mode', help='source_mode: file or screenshot')
    parser.add_argument('--detection_mode', help='detection_mode: default')
    parser.add_argument('--logging', help='logging: False')
    args = parser.parse_args()

    host = args.host if args.host else os.environ.get('HOST', '127.0.0.1')
    port = int(args.port) if args.port else int(os.environ.get('PORT', '8900'))
    source_mode = args.source_mode if args.source_mode else os.environ.get('SOURCE_MODE', 'file')
    detection_mode = args.detection_mode if args.detection_mode else os.environ.get('DETECTION_MODE', 'default')
    logging = args.logging if args.logging else os.environ.get('LOGGING', 'False')

    # Config file should be in the root project folder. Ex: "E:\\Karaburma\\config.json"
    config_path = os.path.join(files_helper.get_project_root_path(), "config.json")

    karaburma = Karaburma(config_path, source_mode, detection_mode, logging)

    screenshot_service = ScreenshotService(karaburma)
    screenshot_controller = ScreenshotController(screenshot_service)

    file_service = FileService(karaburma)
    file_controller = FileController(file_service)

    base64image_service = Base64Service(karaburma)
    base64image_controller = Base64Controller(base64image_service)

    karaburma_api_service = KaraburmaApiService(host, int(port), config_path, source_mode, detection_mode, logging)

    karaburma_api_service.get_app().include_router(root_router)
    karaburma_api_service.get_app().include_router(screenshot_controller.get_screenshot_router())
    karaburma_api_service.get_app().include_router(file_controller.get_file_router())
    karaburma_api_service.get_app().include_router(base64image_controller.get_base64image_router())

    # 'asyncio.run' - is a recommended root point for start application
    asyncio.run(karaburma_api_service.start_karaburma_service())
