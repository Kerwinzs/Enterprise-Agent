"""Minimal OpenAI-compatible Chat Completions transport; easily replaced in tests."""

import json
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.config import Settings


class ProviderError(Exception):
    """A safe public error; never include upstream response bodies or credentials."""


class FunctionCall(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    arguments: str = Field(max_length=10000)


class ToolCall(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    type: Literal["function"] = "function"
    function: FunctionCall


class AssistantTurn(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: Literal["assistant"] = "assistant"
    content: str | None = Field(default=None, max_length=32000)
    tool_calls: list[ToolCall] = Field(default_factory=list, max_length=32)


class ChatCompletionClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient):
        self.settings, self.client = settings, client

    async def complete(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        try:
            response = await self.client.post(
                self.settings.openai_base_url + "/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.settings.openai_api_key.get_secret_value()}"
                },
                json={
                    "model": self.settings.openai_model,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto",
                    "store": False,
                },
                timeout=self.settings.llm_timeout_seconds,
            )
            response.raise_for_status()
            message = response.json()["choices"][0]["message"]
            if not isinstance(message, dict):
                raise ValueError("invalid message")
            # Some compatible providers return null rather than omitting tool_calls.
            if message.get("tool_calls") is None:
                message["tool_calls"] = []
            turn = AssistantTurn.model_validate(message)
            if not turn.tool_calls and not (turn.content or "").strip():
                raise ValueError("empty assistant response")
            ids = [call.id for call in turn.tool_calls]
            if len(ids) != len(set(ids)):
                raise ValueError("duplicate tool call ids")
            return turn
        except httpx.TimeoutException as exc:
            raise ProviderError("大模型接口超时，请稍后重试。") from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in {401, 403}:
                message = "大模型授权失败，请检查服务端密钥和模型权限。"
            elif status == 429:
                message = "大模型额度不足或请求过于频繁，请检查账户后重试。"
            else:
                message = "大模型服务暂不可用，请检查模型与 API 地址配置。"
            raise ProviderError(message) from exc
        except (
            httpx.RequestError,
            ValidationError,
            ValueError,
            KeyError,
            IndexError,
            TypeError,
            json.JSONDecodeError,
        ) as exc:
            raise ProviderError("大模型连接失败或响应格式不兼容，请检查服务端配置。") from exc
