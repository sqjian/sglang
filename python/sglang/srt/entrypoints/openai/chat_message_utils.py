from __future__ import annotations

from typing import Any, Dict

import orjson


def normalize_tool_content(role: str, content: Any) -> Any:
    """Normalize OpenAI text-only tool content parts to a plain string.

    Lists containing non-text parts are preserved because some chat templates
    intentionally iterate over them.
    """
    if role != "tool" or not isinstance(content, list):
        return content

    is_openai_text_parts = all(
        (isinstance(part, dict) and part.get("type") == "text") or isinstance(part, str)
        for part in content
    )
    if not is_openai_text_parts:
        return content

    return " ".join(
        part.get("text", "") if isinstance(part, dict) else part for part in content
    )


def parse_tool_call_arguments(arguments: str) -> Dict[str, Any]:
    """Parse OpenAI tool call arguments for chat templates."""
    try:
        parsed_arguments = orjson.loads(arguments)
    except orjson.JSONDecodeError as exc:
        raise ValueError(
            "Assistant tool call function.arguments must be valid JSON."
        ) from exc

    if not isinstance(parsed_arguments, dict):
        raise ValueError(
            "Assistant tool call function.arguments must be a JSON object."
        )

    return parsed_arguments


def normalize_assistant_tool_call_arguments(
    message: Dict[str, Any], *, strict: bool = True
) -> None:
    """Normalize assistant history tool call arguments in-place."""
    if message.get("role") != "assistant" or not isinstance(
        message.get("tool_calls"), list
    ):
        return

    for item in message["tool_calls"]:
        function = item.get("function") if isinstance(item, dict) else None
        if not isinstance(function, dict):
            continue
        if "arguments" in function and isinstance(function["arguments"], str):
            try:
                function["arguments"] = parse_tool_call_arguments(function["arguments"])
            except ValueError:
                if strict:
                    raise
