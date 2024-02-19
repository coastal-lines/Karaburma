import argparse
import asyncio
import os
import time
import uvicorn
import requests
from fastapi import FastAPI, status
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from starlette.responses import JSONResponse
from karaburma.utils import files_helper

from karaburma.main import Karaburma


class RequestParams(BaseModel):
    image_base64: str
    image_file_path: str
    image_pattern_file_path: str
    type_element: str


class KaraburmaApiService:
    def __init__(self, host, port, config_path, source_mode, detection_mode, logging):

        self._server = None

        self._karaburma_instance = Karaburma(config_path, source_mode, detection_mode, logging)
        self._app = FastAPI()

        self._host = host
        self._port = port

        # Endpoint
        @self._app.get("/", status_code=status.HTTP_200_OK)
        async def root_selfcheck():
            return JSONResponse(
                content={"message": "Uvicorn server was started for Karaburma."},
            )

        # Endpoint
        @self._app.post("/api/v1/file/all_elements", status_code=status.HTTP_200_OK)
        def user_file_find_all_possible_elements(request_params: RequestParams):
            image_file_path = request_params.image_file_path

            result_json = self._karaburma_instance.find_all_elements(image_file_path)
            return result_json

        # Endpoint
        @self._app.post("/api/v1/file/", status_code=status.HTTP_200_OK)
        def user_file_find_selected_element(request_params: RequestParams):
            image_file_path = request_params.image_file_path
            type_element = request_params.type_element

            result_json = self._karaburma_instance.find_element(type_element, image_file_path)
            return result_json

        # Endpoint
        @self._app.post("/api/v1/file/pattern", status_code=status.HTTP_200_OK)
        def user_file_find_by_pattern(request_params: RequestParams):
            image_file_path = request_params.image_file_path
            type_element = request_params.type_element
            image_pattern_file_path = request_params.image_pattern_file_path

            result_json = self._karaburma_instance.find_element_by_patterns(
                [image_pattern_file_path],
                "normal",
                0.8,
                type_element,
                image_file_path)

            return result_json

        # Endpoint
        @self._app.post("/api/v1/file/pattern_all_elements", status_code=status.HTTP_200_OK)
        def user_file_find_by_pattern(request_params: RequestParams):
            image_file_path = request_params.image_file_path
            type_element = request_params.type_element
            image_pattern_file_path = request_params.image_pattern_file_path

            result_json = self._karaburma_instance.find_all_elements_include_patterns(
                [image_pattern_file_path],
                "normal",
                0.8,
                type_element,
                image_file_path)

            return result_json

        # Endpoint
        @self._app.post("/api/v1/screenshot/all_elements", status_code=status.HTTP_200_OK)
        def user_screenshot_find_all_possible_elements():
            result_json = self._karaburma_instance.find_all_elements()
            return result_json

        # Endpoint
        @self._app.post("/api/v1/screenshot/", status_code=status.HTTP_200_OK)
        def user_screenshot_find_selected_element(request_params: RequestParams):
            type_element = request_params.type_element
            result_json = self._karaburma_instance.find_element(type_element)
            return result_json

        # Endpoint
        @self._app.post("/api/v1/screenshot/expand_table", status_code=status.HTTP_200_OK)
        def user_screenshot_find_table_and_expand(request_params: RequestParams):
            result_json = self._karaburma_instance.find_table_and_expand(0)
            return result_json

        # Endpoint
        @self._app.post("/api/v1/screenshot/expand_listbox", status_code=status.HTTP_200_OK)
        def user_screenshot_find_listbox_and_expand(request_params: RequestParams):
            result_json = self._karaburma_instance.find_listbox_and_expand(0)
            return result_json

        # Handler for RequestValidationError
        @self._app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: RequestParams, exc: RequestValidationError):
            return JSONResponse(
                status_code=422,
                content={"message": "Please check json values for your POST request.", "details": exc.errors()}
            )

        # Handler for common exceptions
        @self._app.exception_handler(Exception)
        async def generic_exception_handler(request: RequestParams, exc: Exception):
            return JSONResponse(
                status_code=500,
                content={"message": "Internal server error", "details": str(exc)}
            )

    def start_karaburma_service(self):
        uvicorn_config = uvicorn.Config(app=self._app, host=self._host, port=self._port)
        self._server = uvicorn.Server(uvicorn_config)
        asyncio.run(self._server.serve())

    def check_server_availability(self):
        url = "http://127.0.0.1:8000"
        max_retries = 10
        retries = 0
        wait_time = 2

        while retries < max_retries:
            try:
                response = requests.get(url)
                response.raise_for_status()
                print("Server is ready to accept requests")
                return True
            except requests.RequestException as e:
                print(f"Attempt {retries + 1} failed: {e}")
                retries += 1
                time.sleep(wait_time)

        print("Server did not start within the expected time")
        return False

    def stop_karaburma_service(self):
        self._server.should_exit = True
        self._server.close()

if __name__ == "__main__":
    # Create command line parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('--host', help='host', required=True)
    parser.add_argument('--port', help='port', required=True)
    #parser.add_argument('--config_path', help='config_path', required=True)
    parser.add_argument('--source_mode', help='source_mode: file or screenshot', required=True)
    parser.add_argument('--detection_mode', help='detection_mode: default', required=True)
    parser.add_argument('--logging', help='logging: False', required=True)

    # Parsing arguments of command line
    args = parser.parse_args()

    config_path = os.path.join(files_helper.get_project_root_path(), "config.json")
    #k = KaraburmaApiService("127.0.0.1", 8900, config_path, "file", detection_mode="default", logging=False)
    #k = KaraburmaApiService("127.0.0.1", 8900, config_path, "screenshot", detection_mode="default", logging=False)
    karaburma = KaraburmaApiService(args.host, int(args.port), config_path, args.source_mode, args.detection_mode, args.logging)
    karaburma.start_karaburma_service()