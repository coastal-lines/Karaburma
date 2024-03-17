from fastapi import APIRouter, status
from starlette.responses import JSONResponse

from karaburma.api.schemas.request_models import FileElementRequest, FileImagePatternElementRequest
from karaburma.data.constants.enums.element_types_enum import ElementTypesEnum


class FileController:
    def __init__(self, file_service):
        self.__file_service = file_service
        self.__file_router = APIRouter(prefix="/api/v1/file", tags=["file"])
        self.__file_router.post("/", status_code=status.HTTP_200_OK)(self.user_file_find_element)
        self.__file_router.post("/image_pattern", status_code=status.HTTP_200_OK)(self.user_file_find_element_and_pattern)

    def get_file_router(self):
        return self.__file_router

    async def user_file_find_element(self, request_params: FileElementRequest):
        result_json = dict()

        type_element = request_params.type_element
        is_read_text = request_params.is_read_text
        image_file_path = request_params.image_file_path

        if (type_element not in ElementTypesEnum.__members__):
            return JSONResponse(status_code=400, content={"message": f"'{type_element}' element type is not supported."})

        result_json = self.__file_service.find_element_in_file(type_element, is_read_text, image_file_path)

        return result_json

    # Endpoint http://127.0.0.1:8900/api/v1/file/image_pattern
    async def user_file_find_element_and_pattern(self, request_params: FileImagePatternElementRequest):
        result_json = dict()

        image_pattern_type_element = request_params.image_pattern_type_element
        image_file_path = request_params.image_file_path
        image_pattern_file_path = request_params.image_pattern_file_path
        is_all_elements = request_params.is_all_elements
        search_mode = request_params.search_mode

        result_json = self.__file_service.find_element_by_pattern(image_pattern_type_element,
                                                                  image_file_path,
                                                                  image_pattern_file_path,
                                                                  is_all_elements,
                                                                  search_mode)

        return result_json

