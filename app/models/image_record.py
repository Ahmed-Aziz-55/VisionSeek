from pydantic import BaseModel, field_validator

class ImageRecord(BaseModel):
    image_path: str            # original raw image path
    caption: str
    category: str
    processed_path: str | None = None   # path to resized/normalized image on disk

    @field_validator("caption", "category")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()