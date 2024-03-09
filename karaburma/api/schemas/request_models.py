from pydantic import BaseModel


class ScreenshotElementRequest(BaseModel):
    type_element: str
    is_fully_expanded: bool
    is_read_text: bool


class ScreenshotTableElementRequest(BaseModel):
    table_number: int


class FileElementRequest(BaseModel):
    image_file_path: str
    type_element: str
    is_read_text: bool


class FileImagePatternElementRequest(BaseModel):
    image_file_path: str
    image_pattern_file_path: str
    image_pattern_type_element: str
    is_all_elements: bool


class Base64ElementRequest(BaseModel):
    base64_image: str
    type_element: str
    is_read_text: bool


class Base64PatternElementRequest(BaseModel):
    base64_image: str
    image_pattern_base64_image: str
    image_pattern_type_element: str