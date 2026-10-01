from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class AnalyzeRequest(BaseModel):
    text: str | None = Field(default=None, max_length=20000)
    url: str | None = Field(default=None, max_length=2048)
    mode: Literal["fast", "deep"] = "deep"

    @model_validator(mode="after")
    def _one_of(self):
        if not (self.text and self.text.strip()) and not (self.url and self.url.strip()):
            raise ValueError("provide `text` or `url`")
        return self


class FeedbackRequest(BaseModel):
    analysis_id: str
    rating: Literal["correct", "incorrect"]
    comment: str | None = Field(default=None, max_length=1000)


class BatchRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=200)
