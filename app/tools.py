import json
from typing import Literal

from pydantic import Field, ValidationError

from app.database import OrderStore
from app.retrieval import DocumentIndex
from app.schemas import StrictModel


class SearchArguments(StrictModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=3, ge=1, le=5)


class OrderArguments(StrictModel):
    status: Literal["all", "paid", "unpaid", "overdue"] = "all"
    customer: str | None = Field(default=None, min_length=1, max_length=80)
    limit: int = Field(default=10, ge=1, le=50)


ARGUMENT_MODELS = {"search_documents": SearchArguments, "query_orders": OrderArguments}
DESCRIPTIONS = {
    "search_documents": "检索企业制度文档。返回带 id 的原文片段，回答时用 [id] 引用。",
    "query_orders": (
        "只读查询虚构订单快照。status=unpaid 包含 unpaid 和 overdue；"
        "customer 为精确客户名或 null；金额单位为分；汇总覆盖所有匹配行，列表受 limit 限制。"
        "仅支持状态与客户筛选，不支持日期、部门等条件，不得假装这些条件已生效。"
    ),
}


def tool_definitions() -> list[dict]:
    definitions = []
    for name, model in ARGUMENT_MODELS.items():
        schema = model.model_json_schema()
        # Strict function schemas require every property, including nullable ones, to be required.
        schema["required"] = list(schema["properties"])
        for prop in schema["properties"].values():
            prop.pop("default", None)
        definitions.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": DESCRIPTIONS[name],
                    "parameters": schema,
                    "strict": True,
                },
            }
        )
    return definitions


class ToolRegistry:
    def __init__(self, index: DocumentIndex, orders: OrderStore):
        self.index, self.orders = index, orders

    def execute(self, name: str, raw_arguments: str) -> tuple[dict, dict]:
        if name not in ARGUMENT_MODELS:
            return {}, {"error": "unknown_tool", "message": "工具未在允许列表中"}
        if len(raw_arguments) > 4000:
            return {}, {"error": "invalid_arguments", "message": "工具参数过长"}
        try:
            args = ARGUMENT_MODELS[name].model_validate_json(raw_arguments, strict=True)
        except ValidationError:
            return {}, {"error": "invalid_arguments", "message": "参数必须符合工具 JSON schema"}
        arguments = args.model_dump()
        if name == "search_documents":
            matches = self.index.search(**arguments)
            return arguments, {"matches": [match.model_dump() for match in matches]}
        return arguments, self.orders.query(**arguments)

    def invoke(self, name: str, arguments: dict) -> tuple[dict, dict]:
        return self.execute(name, json.dumps(arguments, ensure_ascii=False))
