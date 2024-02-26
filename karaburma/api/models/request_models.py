from pydantic import BaseModel


class ScreenshotElementRequest(BaseModel):
    type_element: str
    is_fully_expanded: bool

class FileElementRequest(BaseModel):
    image_file_path: str
    type_element: str

class FileImagePatternElementRequest(BaseModel):
    image_file_path: str
    image_pattern_file_path: str
    image_pattern_type_element: str
    is_all_elements: bool
