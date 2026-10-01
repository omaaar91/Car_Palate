from typing import List, Optional
from pydantic import BaseModel

class CharBoxDetail(BaseModel):
    box: List[int]
    class_name: Optional[str] = None
    arabic: Optional[str] = None
    is_digit: bool

class PlateResult(BaseModel):
    text: str
    digits: List[str]
    letters: List[str]
    confidence: float
    bbox: List[int]
    angle: float
    track_id: Optional[int] = None
    char_details: Optional[List[CharBoxDetail]] = None
    syntax_valid: Optional[bool] = None
    governorate: Optional[str] = None
    format_code: Optional[str] = None
    badge: Optional[str] = None

class ALPRResponse(BaseModel):
    success: bool
    plates_count: int
    plates: List[PlateResult]
    annotated_image_base64: Optional[str] = None
    message: str = "Success"
