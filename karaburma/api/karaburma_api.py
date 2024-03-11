import argparse
import asyncio
import os
import traceback
import uvicorn
from http.client import HTTPException
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse

from karaburma.data.constants.enums.element_types_enum import ElementTypesEnum
from karaburma.api.schemas.request_models import ScreenshotElementRequest, FileElementRequest, \
    FileImagePatternElementRequest, ScreenshotTableElementRequest, Base64ElementRequest, Base64PatternElementRequest
from karaburma.utils import files_helper
from karaburma.main import Karaburma


class KaraburmaApiService:
    def __init__(self, host, port, config_path, source_mode, detection_mode, logging):
        self.__server = None
        self.__server_task = None

        self.__karaburma_instance = Karaburma(config_path, source_mode, detection_mode, logging)
        self.__app = FastAPI()

        self.__host = host
        self.__port = port

        # Router
        # Endpoint http://127.0.0.1:8900/
        @self.__app.get("/", status_code=status.HTTP_200_OK)
        async def root_selfcheck():
            return JSONResponse(content={"message": "Uvicorn server was started for Karaburma."})

        # Router
        # Endpoint http://127.0.0.1:8900/api/v1/base64image
        @self.__app.post("/api/v1/base64image/", status_code=status.HTTP_200_OK)
        async def user_base64image_find_element(request_params: Base64ElementRequest):
            result_json = dict()

            base64_image = request_params.base64_image
            type_element = request_params.type_element
            is_read_text = request_params.is_read_text

            user_image = files_helper.base64_to_image(base64_image)

            match type_element:
                case "all":
                    if is_read_text:
                        result_json = self.__karaburma_instance.find_all_elements_in_base64image(user_image, True)
                    else:
                        result_json = self.__karaburma_instance.find_all_elements_in_base64image(user_image)
                case _:
                    if (type_element not in ElementTypesEnum.__members__):
                        return JSONResponse(status_code=400,
                                            content={"message": f"'{type_element}' element type is not supported."})

                    result_json = self.__karaburma_instance.find_element_in_base64image(type_element, user_image)

            return result_json

        # Router
        # Endpoint http://127.0.0.1:8900/api/v1/base64image/image_pattern
        @self.__app.post("/api/v1/base64image/image_pattern", status_code=status.HTTP_200_OK)
        async def user_base64image_find_element(request_params: Base64PatternElementRequest):
            result_json = dict()

            base64_image = request_params.base64_image
            image_pattern_base64_image = request_params.image_pattern_base64_image
            image_pattern_type_element = request_params.image_pattern_type_element

            user_image = files_helper.base64_to_image(base64_image)
            image_pattern = files_helper.base64_to_image(image_pattern_base64_image)

            result_json = self.__karaburma_instance.find_all_elements_include_patterns_in_base64image(
                image_pattern,
                "normal",
                0.8,
                image_pattern_type_element,
                user_image
            )

            return result_json

        # Router
        # Endpoint http://127.0.0.1:8900/api/v1/file/
        @self.__app.post("/api/v1/file/", status_code=status.HTTP_200_OK)
        async def user_file_find_element(request_params: FileElementRequest):
            result_json = dict()

            image_file_path = request_params.image_file_path
            type_element = request_params.type_element
            is_read_text = request_params.is_read_text

            match type_element:
                case "all":
                    if is_read_text:
                        result_json = self.__karaburma_instance.find_all_elements_and_read_text(image_file_path)
                    else:
                        result_json = self.__karaburma_instance.find_all_elements(image_file_path)
                case _:
                    if (type_element not in ElementTypesEnum.__members__):
                        return JSONResponse(status_code=400,
                                            content={"message": f"'{type_element}' element type is not supported."})

                    result_json = self.__karaburma_instance.find_element(type_element, image_file_path)

            return result_json

        # Router
        # Endpoint http://127.0.0.1:8900/api/v1/file/image_pattern
        @self.__app.post("/api/v1/file/image_pattern", status_code=status.HTTP_200_OK)
        async def user_file_find_element(request_params: FileImagePatternElementRequest):
            result_json = dict()

            image_pattern_type_element = request_params.image_pattern_type_element
            image_file_path = request_params.image_file_path
            image_pattern_file_path = request_params.image_pattern_file_path
            is_all_elements = request_params.is_all_elements

            if (is_all_elements):
                result_json = self.__karaburma_instance.find_all_elements_include_patterns(
                    [image_pattern_file_path],
                    "normal",
                    0.8,
                    image_pattern_type_element,
                    image_file_path)
            else:
                result_json = self.__karaburma_instance.find_element_by_patterns(
                    [image_pattern_file_path],
                    "normal",
                    0.8,
                    image_pattern_type_element,
                    image_file_path)

            return result_json

        # Router
        # Endpoint http://127.0.0.1:8900/api/v1/screenshot
        @self.__app.post("/api/v1/screenshot/", status_code=status.HTTP_200_OK)
        async def user_screenshot_find_element(request_params: ScreenshotElementRequest):
            result_json = dict()

            type_element = request_params.type_element
            is_fully_expanded = request_params.is_fully_expanded
            is_read_text = request_params.is_read_text

            match type_element:
                case ElementTypesEnum.table.name:
                    if is_fully_expanded:
                        result_json = self.__karaburma_instance.find_table_and_expand(0)
                case ElementTypesEnum.listbox.name:
                    if is_fully_expanded:
                        result_json = self.__karaburma_instance.find_listbox_and_expand(0)
                    else:
                        result_json = self.__karaburma_instance.find_element(type_element)
                case "all":
                    if is_read_text:
                        result_json = self.__karaburma_instance.find_all_elements_and_read_text()
                    else:
                        result_json = self.__karaburma_instance.find_all_elements()
                case _:
                    if type_element not in ElementTypesEnum.__members__:
                        return JSONResponse( status_code=400,
                            content={"message": f"'{type_element}' element type is not supported."}
                        )

                    result_json = self.__karaburma_instance.find_element(type_element)

            return result_json

        # Router
        # Endpoint http://127.0.0.1:8900/api/v1/screenshot/table_with_text
        @self.__app.post("/api/v1/screenshot/table_with_text", status_code=status.HTTP_200_OK)
        async def user_screenshot_get_text_from_table(request_params: ScreenshotTableElementRequest):
            table_number = request_params.table_number
            return self.__karaburma_instance.find_table_and_expand(table_number, True)

        @self.__app.exception_handler(400)
        async def not_found_exception_handler(request: Request, exc: HTTPException):
            return JSONResponse(status_code=400, content={"message": f"Please check ."})

        @self.__app.exception_handler(404)
        async def not_found_exception_handler(request: Request, exc: HTTPException):
            return JSONResponse(status_code=404, content={"message": f"Url '{str(request.url)}' not found."})

        @self.__app.exception_handler(405)
        async def not_found_exception_handler(request: Request, exc: HTTPException):
            return JSONResponse(status_code=405, content={"message": f"Url '{str(request.url)}' not found!!!!!!!!!!!!."})

        # Handler for RequestValidationError
        # FastApi has built-in handle for validating Pydantic model.
        # Pydantic is a data validation and settings management library in Python.
        @self.__app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, exc: RequestValidationError):
            return JSONResponse(status_code=422,
                content={"message": "Please check json values for your POST request.", "details": exc.errors()}
            )

        # Handler for server exceptions
        @self.__app.exception_handler(Exception)
        async def generic_exception_handler(request: Request, exc: Exception):
            traceback_str = "".join(traceback.format_tb(exc.__traceback__))
            return JSONResponse(
                status_code=500,
                content={"message": "Internal server error",
                         "error": str(exc),
                         "traceback": f"{exc}\n\n{traceback_str}"
                         }
            )

    def get_app(self):
        return self.__app

    # For starting Karaburma by command line
    async def start_karaburma_service(self):
        uvicorn_config = uvicorn.Config(app=self.__app, host=self.__host, port=self.__port)
        self.__server = uvicorn.Server(uvicorn_config)
        await self.__server.serve()

    # For starting Karaburma by test fixtures
    def start_karaburma_service_for_test_fixtures(self):
        uvicorn_config = uvicorn.Config(app=self.__app, host=self.__host, port=self.__port, log_level="trace")
        self.__server = uvicorn.Server(uvicorn_config)
        task = asyncio.create_task(self.__server.serve())

        return task

    def start_karaburma_service_for_test_fixtures_2(self):
        uvicorn_config = uvicorn.Config(app=self.__app, host=self.__host, port=self.__port, log_level="trace")
        self.__server = uvicorn.Server(uvicorn_config)
        self.__server.serve()

    async def start_karaburma_service_for_test_fixtures_3(self):
        uvicorn_config = uvicorn.Config(app=self.__app, host=self.__host, port=self.__port, log_level="trace")
        self.__server = uvicorn.Server(uvicorn_config)
        self.__server_task = asyncio.create_task(self.__server.serve())
        await self.__server_task

    async def stop_karaburma_service(self):
        if self.__server is not None:
            self.__server.should_exit = True
            # Wait few seconds for stop
            await asyncio.sleep(2)
            self.__server = None

    async def stop_karaburma_service2(self):
        if self.__server is not None:
            await self.__server.shutdown()
            await self.__server.wait_closed()

if __name__ == "__main__":
    # Create command line parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('--host', help='host', required=True)
    parser.add_argument('--port', help='port', required=True)
    parser.add_argument('--source_mode', help='source_mode: file or screenshot', required=True)
    parser.add_argument('--detection_mode', help='detection_mode: default', required=True)
    parser.add_argument('--logging', help='logging: False', required=True)

    # Parsing arguments of command line
    args = parser.parse_args()

    # Config file should be in the root project folder. Ex: "E:\\Karaburma\\config.json"
    config_path = os.path.join(files_helper.get_project_root_path(), "config.json")
    karaburma = KaraburmaApiService(args.host, int(args.port), config_path, args.source_mode, args.detection_mode, args.logging)

    # 'asyncio.run' - is a recommended root start point for application
    asyncio.run(karaburma.start_karaburma_service())