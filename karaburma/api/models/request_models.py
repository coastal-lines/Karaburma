from pydantic import BaseModel

class ScreenshotCustomElementRequest(BaseModel):
    type_element: str

class ScreenshotElementRequest(BaseModel):
    type_element: str
    is_fully_expanded: bool

class RequestParams(BaseModel):
    image_base64: str
    image_file_path: str
    image_pattern_file_path: str
    type_element: str