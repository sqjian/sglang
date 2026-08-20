import copy
import json
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from transformers import PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import BaseDataset, DatasetRow
from sglang.srt.entrypoints.openai.chat_message_utils import (
    normalize_assistant_tool_call_arguments,
    normalize_tool_content,
)


@dataclass
class OpenAIDataset(BaseDataset):
    dataset_path: str
    num_requests: int
    fixed_output_len: Optional[int]
    extra_request_body: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_args(cls, args: Namespace) -> "OpenAIDataset":
        raw_extra_request_body = getattr(args, "extra_request_body", None)
        extra_request_body = (
            json.loads(raw_extra_request_body) if raw_extra_request_body else {}
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


def _normalize_messages_for_tokenizer(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    normalized_messages = copy.deepcopy(messages)
    for message in normalized_messages:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise ValueError("OpenAI dataset message must contain a string role")
        if message.get("content") is None:
            message["content"] = ""
        message["content"] = normalize_tool_content(
            message["role"], message.get("content")
        )
        normalize_assistant_tool_call_arguments(message)
    return normalized_messages


def _effective_request_body(
    record_body: Dict[str, Any], global_body: Dict[str, Any]
) -> Dict[str, Any]:
    # Match benchmark(): per-request dataset fields override global CLI fields.
    return {**copy.deepcopy(global_body), **copy.deepcopy(record_body)}


def _render_prompt(
    tokenizer: PreTrainedTokenizerBase,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    template_kwargs: Dict[str, Any],
) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            return_dict=False,
            **template_kwargs,
        )
    except Exception:
        if not tools:
            raise
        flat_tools = [tool.get("function", tool) for tool in tools]
        return tokenizer.apply_chat_template(
            messages,
            tools=flat_tools,
            tokenize=False,
            add_generation_prompt=True,
            return_dict=False,
            **template_kwargs,
        )


def _count_prompt_tokens(
    tokenizer: PreTrainedTokenizerBase,
    messages: List[Dict[str, Any]],
    extra_body: Dict[str, Any],
) -> int:
    normalized_messages = _normalize_messages_for_tokenizer(messages)
    template_kwargs = copy.deepcopy(extra_body.get("chat_template_kwargs", {}))
    if not isinstance(template_kwargs, dict):
        raise ValueError("chat_template_kwargs must be an object")
    tools = copy.deepcopy(extra_body.get("tools"))
    if tools is not None and not isinstance(tools, list):
        raise ValueError("tools must be an array")

    rendered_prompt = _render_prompt(
        tokenizer=tokenizer,
        messages=normalized_messages,
        tools=tools,
        template_kwargs=template_kwargs,
    )
    return len(tokenizer.encode(rendered_prompt, add_special_tokens=False))


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
    global_body = {} if extra_request_body is None else extra_request_body
    if not isinstance(global_body, dict):
        raise ValueError("extra_request_body must be an object")

    dataset = []
    with open(dataset_path, "r") as f:
        for line in f:
            if num_requests > 0 and len(dataset) >= num_requests:
                break
            if line.strip():
                try:
                    dataset.append(json.loads(line))
                except json.JSONDecodeError:
                    # Skip invalid JSON lines
                    continue

    # Fields that should NOT be passed through extra_request_body
    # These are either handled separately or are metadata
    # max_tokens is excluded because it's handled via output_len -> max_completion_tokens
    # max_completion_tokens is also excluded to avoid conflicts
    EXCLUDED_FIELDS = {"messages", "max_tokens", "max_completion_tokens", "model"}

    filtered_dataset: List[DatasetRow] = []
    for data in dataset:
        messages = data.get("messages", [])
        if not messages:
            continue

        # Use max_tokens from the request, or fall back to fixed_output_len
        output_len = fixed_output_len or data.get("max_tokens", 256)

        # Extract extra request body parameters (tools, temperature, top_p, etc.)
        extra_body = {k: v for k, v in data.items() if k not in EXCLUDED_FIELDS}
        effective_body = _effective_request_body(extra_body, global_body)
        prompt_len = _count_prompt_tokens(tokenizer, messages, effective_body)

        # Pass messages list directly - the serving benchmark handles List[Dict] prompts
        filtered_dataset.append(
            DatasetRow(
                prompt=messages,
                prompt_len=prompt_len,
                output_len=output_len,
                extra_request_body=extra_body,  # Store per-request parameters
            )
        )

    print(f"Loaded {len(filtered_dataset)} OpenAI-format requests")
    print(f"#Input tokens: {np.sum([x.prompt_len for x in filtered_dataset])}")
    print(f"#Output tokens: {np.sum([x.output_len for x in filtered_dataset])}")
    return filtered_dataset
