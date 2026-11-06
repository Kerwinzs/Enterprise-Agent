from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HistoryMessage(StrictModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(StrictModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=12)

    @field_validator("message")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("请输入问题")
        return value.strip()


class Source(BaseModel):
    id: str
    document: str
    title: str
    text: str
    score: float


class ToolTrace(BaseModel):
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    elapsed_ms: float


class ChatResponse(BaseModel):
    request_id: str
    mode: Literal["demo", "live"]
    answer: str
    sources: list[Source]
    tool_calls: list[ToolTrace]
    elapsed_ms: float
