from pydantic import BaseModel, Field
from typing import ClassVar, Optional
from datetime import datetime

class Resource(BaseModel):
    type: ClassVar[str] = "resource"

class PDFDocument(Resource):
    type: ClassVar[str] = "pdf"
    source_url: Optional[str] = None
    title: Optional[str] = None
    fetched_at: datetime = Field(default_factory=datetime.now)


class YouTubeVideo(Resource):
    type: ClassVar[str] = "youtube"
    title: Optional[str] = None
    fetched_at: datetime = Field(default_factory=datetime.now)
