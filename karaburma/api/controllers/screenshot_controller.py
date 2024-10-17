from fastapi import APIRouter, status
from starlette.responses import JSONResponse

from karaburma.api.schemas.request_models import ScreenshotElementRequest, ScreenshotTableElementRequest
from karaburma.data.constants.enums.element_types_enum import ElementTypesEnum


class ScreenshotController:
    def __init__(self, screenshot_service):
        self.__screenshot_service = screenshot_service
        self.__screenshot_router = APIRouter(prefix="/api/v1/screenshot", tags=["screenshot"])
        self.__screenshot_router.post("/", status_code=status.HTTP_200_OK)(self.user_screenshot_find_element)
        self.__screenshot_router.post("/table_with_text", status_code=status.HTTP_200_OK)(self.user_screenshot_get_text_from_table)

    def get_screenshot_router(self):
        return self.__screenshot_router

    async def user_screenshot_find_element(self, request_params: ScreenshotElementRequest):
        result_json = dict()

        type_element = request_params.type_element
        is_fully_expanded = request_params.is_fully_expanded
        is_read_text = request_params.is_read_text

        if type_element not in ElementTypesEnum.__members__:
            return JSONResponse(status_code=400, content={"message": f"'{type_element}' element type is not supported."})

        result_json = self.__screenshot_service.find_elements_in_screenshot(type_element, is_fully_expanded, is_read_text)

        return result_json

    # Endpoint http://127.0.0.1:8900/api/v1/screenshot/table_with_text
    async def user_screenshot_get_text_from_table(self, request_params: ScreenshotTableElementRequest):
        result_json = dict()
        result_json = self.__screenshot_service.find_text_with_expanded_table_in_screenshot(request_params.table_number)
        return result_json




