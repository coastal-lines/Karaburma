from fastapi import APIRouter, status
from starlette.responses import JSONResponse

from karaburma.api.schemas.request_models import Base64ElementRequest, Base64PatternElementRequest
from karaburma.data.constants.enums.element_types_enum import ElementTypesEnum


class Base64Controller:
    def __init__(self, base64_service):
        self.__base64_service = base64_service
        self.__base64_router = APIRouter(prefix="/api/v1/base64image", tags=["base64"])
        self.__base64_router.post("/", status_code=status.HTTP_200_OK)(self.user_base64image_find_element)
        self.__base64_router.post("/image_pattern", status_code=status.HTTP_200_OK)(self.user_base64image_find_element_and_pattern)

    def get_base64image_router(self):
        return self.__base64_router

    # Endpoint http://127.0.0.1:8900/api/v1/base64image
    async def user_base64image_find_element(self, request_params: Base64ElementRequest):
        result_json = dict()

        type_element = request_params.type_element
        is_read_text = request_params.is_read_text
        base64_image = request_params.base64_image

        if (type_element not in ElementTypesEnum.__members__):
            return JSONResponse(status_code=400, content={"message": f"'{type_element}' element type is not supported."})

        result_json = self.__base64_service.find_element_in_base64image(type_element, is_read_text, base64_image)

        return result_json

    # Endpoint http://127.0.0.1:8900/api/v1/base64image/image_pattern
    async def user_base64image_find_element_and_pattern(self, request_params: Base64PatternElementRequest):
        result_json = dict()

        image_pattern_type_element = request_params.image_pattern_type_element
        image_pattern_base64_image = request_params.image_pattern_base64_image
        base64_image = request_params.base64_image

        result_json = self.__base64_service.find_all_elements_and_pattern_in_base64image(image_pattern_type_element,
                                                                                         image_pattern_base64_image,
                                                                                         base64_image)

        return result_json
