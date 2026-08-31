import copy
import json
from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from transformers import PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import BaseDataset, DatasetRow


@dataclass
class OpenAIDataset(BaseDataset):
    dataset_path: str
    num_requests: int
    fixed_output_len: Optional[int]
    extra_request_body: Dict[str, Any]

    @classmethod
    def from_args(cls, args: Namespace) -> "OpenAIDataset":
        extra_request_body = (
            json.loads(args.extra_request_body) if args.extra_request_body else {}
        )
        if not isinstance(extra_request_body, dict):
            raise ValueError("--extra-request-body must be a JSON object")
        return cls(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            fixed_output_len=args.sharegpt_output_len,
            extra_request_body=extra_request_body,
        )

    def load(
        self, tokenizer: PreTrainedTokenizerBase, model_id=None
    ) -> List[DatasetRow]:
        return sample_openai_requests(
            dataset_path=self.dataset_path,
            num_requests=self.num_requests,
            tokenizer=tokenizer,
            fixed_output_len=self.fixed_output_len,
            extra_request_body=self.extra_request_body,
        )


def _normalize_tool_content(role: str, content: Any) -> Any:
    if role != "tool" or not isinstance(content, list):
        return content
    if all(
        (isinstance(part, dict) and part.get("type") == "text") or isinstance(part, str)
        for part in content
    ):
        return " ".join(
            part.get("text", "") if isinstance(part, dict) else part for part in content
        )
    return content


def _normalize_messages_for_tokenizer(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    normalized = copy.deepcopy(messages)
    for message in normalized:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise ValueError("OpenAI dataset message must contain a string role")
        if message.get("content") is None:
            message["content"] = ""
        message["content"] = _normalize_tool_content(
            message["role"], message.get("content")
        )
        if message["role"] != "assistant" or not isinstance(
            message.get("tool_calls"), list
        ):
            continue
        for tool_call in message["tool_calls"]:
            try:
                arguments = tool_call["function"].get("arguments")
            except (KeyError, TypeError, AttributeError) as error:
                raise ValueError("OpenAI dataset tool call is invalid") from error
            if isinstance(arguments, str):
                try:
                    tool_call["function"]["arguments"] = json.loads(arguments)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "OpenAI dataset tool call arguments are invalid JSON"
                    ) from error
    return normalized


def _merge_request_body(
    record_body: Dict[str, Any], global_body: Dict[str, Any]
) -> Dict[str, Any]:
    merged = copy.deepcopy(record_body)
    for key, value in global_body.items():
        if key == "chat_template_kwargs" and isinstance(value, dict):
            row_kwargs = merged.get(key, {})
            if not isinstance(row_kwargs, dict):
                raise ValueError("chat_template_kwargs must be an object")
            merged[key] = {**row_kwargs, **copy.deepcopy(value)}
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _count_prompt_tokens(
    tokenizer: PreTrainedTokenizerBase,
    messages: List[Dict[str, Any]],
    extra_body: Dict[str, Any],
) -> int:
    normalized_messages = _normalize_messages_for_tokenizer(messages)
    chat_template_kwargs = copy.deepcopy(extra_body.get("chat_template_kwargs", {}))
    if not isinstance(chat_template_kwargs, dict):
        raise ValueError("chat_template_kwargs must be an object")
    tools = copy.deepcopy(extra_body.get("tools"))
    template_kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": False,
        "tools": tools,
        **chat_template_kwargs,
    }
    try:
        input_ids = tokenizer.apply_chat_template(
            normalized_messages, **template_kwargs
        )
    except Exception:
        if not tools:
            raise
        template_kwargs["tools"] = [tool.get("function", tool) for tool in tools]
        input_ids = tokenizer.apply_chat_template(
            normalized_messages, **template_kwargs
        )
    if not isinstance(input_ids, list):
        raise ValueError("tokenizer chat template must return token ids")
    return len(input_ids)


def sample_openai_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
    extra_request_body: Optional[Dict[str, Any]] = None,
) -> List[DatasetRow]:
    """
    Load OpenAI-compatible chat completion requests from a JSONL file.

    Each line should be a JSON object with:
    - "messages": list of {"role": str, "content": str}
    - "max_tokens": int (used as output_len if fixed_output_len not set)
    - "tools": optional list of tool definitions
    - "temperature": optional temperature value
    - "top_p": optional top_p value
    - Other OpenAI API parameters are also extracted and passed through
    """
    global_body = extra_request_body or {}
    if not isinstance(global_body, dict):
        raise ValueError("extra_request_body must be an object")
    filtered_dataset: List[DatasetRow] = []
    with open(dataset_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            messages = data.get("messages")
            if not isinstance(messages, list) or not messages:
                continue
            output_len = (
                fixed_output_len
                if fixed_output_len is not None
                else data.get("max_tokens", data.get("max_completion_tokens", 256))
            )
            if (
                isinstance(output_len, bool)
                or not isinstance(output_len, int)
                or output_len < 1
            ):
                raise ValueError(
                    "OpenAI dataset output length must be a positive integer"
                )
            excluded_fields = {
                "messages",
                "max_tokens",
                "max_completion_tokens",
                "model",
            }
            record_body = {
                key: value for key, value in data.items() if key not in excluded_fields
            }
            merged_body = _merge_request_body(record_body, global_body)
            prompt_len = _count_prompt_tokens(tokenizer, messages, merged_body)
            filtered_dataset.append(
                DatasetRow(
                    prompt=messages,
                    prompt_len=prompt_len,
                    output_len=output_len,
                    extra_request_body=merged_body,
                )
            )
            if num_requests > 0 and len(filtered_dataset) >= num_requests:
                break

    print(f"Loaded {len(filtered_dataset)} OpenAI-format requests")
    print(f"#Input tokens: {np.sum([x.prompt_len for x in filtered_dataset])}")
    print(f"#Output tokens: {np.sum([x.output_len for x in filtered_dataset])}")
    return filtered_dataset
