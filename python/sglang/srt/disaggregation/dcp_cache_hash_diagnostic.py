# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Opt-in, content-free cache hashing for PD DCP diagnosis.

The diagnostic emits keyed-XOR SHA-256 digests rather than prompts or tensor
contents. Keying every row by its global token position makes hashes from DCP
partitions composable offline while still detecting slot permutations.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

_ENABLE_ENV = "SGLANG_DEBUG_DCP_CACHE_HASH"
_SEQ_LEN_ENV = "SGLANG_DEBUG_DCP_CACHE_HASH_SEQ_LEN"
_LOG_PREFIX = "DCP_CACHE_HASH_DIAGNOSTIC "
_ZERO_DIGEST = "00" * hashlib.sha256().digest_size
_PREFILL_LAYER_ENABLE_ENV = "SGLANG_DEBUG_PREFILL_LAYER_HASH"
_PREFILL_LAYER_SEQ_LEN_ENV = "SGLANG_DEBUG_PREFILL_LAYER_HASH_SEQ_LEN"
_PREFILL_LAYER_MIN_ENV = "SGLANG_DEBUG_PREFILL_LAYER_HASH_MIN_LAYER"
_PREFILL_LAYER_MAX_ENV = "SGLANG_DEBUG_PREFILL_LAYER_HASH_MAX_LAYER"
_PREFILL_LAYER_IDS_ENV = "SGLANG_DEBUG_PREFILL_LAYER_HASH_LAYERS"
_PREFILL_SUB_LAYER_ENABLE_ENV = "SGLANG_DEBUG_PREFILL_SUB_LAYER_HASH"
_PREFILL_MLP_ENABLE_ENV = "SGLANG_DEBUG_PREFILL_MLP_HASH"
_PREFILL_MLP_ALL_RANK_ENABLE_ENV = "SGLANG_DEBUG_PREFILL_MLP_ALL_RANK_HASH"
_PREFILL_MLP_ALL_RANK_INTERNAL_ENABLE_ENV = (
    "SGLANG_DEBUG_PREFILL_MLP_ALL_RANK_INTERNAL_HASH"
)


@dataclass(frozen=True)
class PrefillLayerHashConfig:
    seq_len: int
    min_layer: int
    max_layer: int
    layer_ids: frozenset[int] | None = None
    log_layer_boundaries: bool = True
    log_sub_layer_boundaries: bool = False
    log_mlp_boundaries: bool = False
    log_all_rank_outer_reduce_boundaries: bool = False
    log_all_rank_mlp_internal_boundaries: bool = False

    def includes(self, layer_id: int) -> bool:
        if self.layer_ids is not None:
            return layer_id in self.layer_ids
        return self.min_layer <= layer_id <= self.max_layer


@dataclass(frozen=True)
class PrefillMlpHashContext:
    rid: str
    seq_len: int
    positions: torch.Tensor


def should_log_cache_hash(seq_len: int) -> bool:
    if os.getenv(_ENABLE_ENV, "0") != "1":
        return False
    raw_target = os.getenv(_SEQ_LEN_ENV)
    if raw_target is None:
        raise ValueError(f"{_SEQ_LEN_ENV} must be set when {_ENABLE_ENV}=1")
    try:
        target = int(raw_target)
    except ValueError as error:
        raise ValueError(
            f"{_SEQ_LEN_ENV} must be an integer, got {raw_target!r}"
        ) from error
    if target <= 0:
        raise ValueError(f"{_SEQ_LEN_ENV} must be positive, got {target}")
    return seq_len == target


def build_owned_slot_plan(
    global_slots: torch.Tensor, *, dcp_size: int, dcp_rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return global positions and rank-local slots owned by one DCP rank."""
    if global_slots.ndim != 1:
        raise ValueError(
            f"global_slots must be one-dimensional, got {global_slots.shape}"
        )
    if dcp_size <= 0 or not 0 <= dcp_rank < dcp_size:
        raise ValueError(f"invalid DCP topology: size={dcp_size}, rank={dcp_rank}")

    slots = global_slots.to(dtype=torch.int64)
    positions = torch.arange(slots.numel(), dtype=torch.int64, device=slots.device)
    if dcp_size == 1:
        return positions, slots

    owned = torch.remainder(slots, dcp_size) == dcp_rank
    return positions[owned], slots[owned] // dcp_size


def hash_int_sequence(values: Iterable[int] | np.ndarray | torch.Tensor) -> str:
    if isinstance(values, torch.Tensor):
        array = values.detach().to(device="cpu", dtype=torch.int64).numpy()
    else:
        array = np.asarray(
            list(values) if not isinstance(values, np.ndarray) else values
        )
        array = array.astype("<i8", copy=False)
    return hashlib.sha256(array.astype("<i8", copy=False).tobytes()).hexdigest()


def xor_hex_digests(digests: Iterable[str]) -> str:
    accumulator = bytearray(hashlib.sha256().digest_size)
    for digest in digests:
        raw = bytes.fromhex(digest)
        if len(raw) != len(accumulator):
            raise ValueError(f"expected a SHA-256 digest, got {digest!r}")
        for index, value in enumerate(raw):
            accumulator[index] ^= value
    return bytes(accumulator).hex()


def hash_positioned_rows(
    rows: torch.Tensor, positions: torch.Tensor, *, layer_id: int
) -> str:
    """Hash rows as a commutative set keyed by layer and global position."""
    if rows.ndim == 0:
        raise ValueError("rows must include a row dimension")
    if rows.shape[0] != positions.numel():
        raise ValueError(
            f"row/position length mismatch: rows={rows.shape[0]}, positions={positions.numel()}"
        )
    if rows.shape[0] == 0:
        return _ZERO_DIGEST

    byte_rows = (
        rows.detach()
        .contiguous()
        .view(torch.uint8)
        .reshape(rows.shape[0], -1)
        .to(device="cpu")
    )
    positions_cpu = positions.detach().to(device="cpu", dtype=torch.int64).tolist()
    accumulator = bytearray(hashlib.sha256().digest_size)
    for position, row in zip(positions_cpu, byte_rows, strict=True):
        digest = hashlib.sha256(
            struct.pack("<qq", int(layer_id), int(position)) + row.numpy().tobytes()
        ).digest()
        for index, value in enumerate(digest):
            accumulator[index] ^= value
    return bytes(accumulator).hex()


def _topology() -> dict[str, int]:
    parallel = get_parallel()
    return {
        "world_rank": int(parallel.world_rank),
        "tp_rank": int(parallel.tp_rank),
        "pp_rank": int(parallel.pp_rank),
        "attn_tp_rank": int(parallel.attn_tp_rank),
        "attn_dp_rank": int(parallel.attn_dp_rank),
        "dcp_size": int(parallel.attn_dcp_size),
        "dcp_rank": int(parallel.attn_dcp_rank),
    }


def _layer_records(
    buffers: Sequence[torch.Tensor],
    slots: torch.Tensor,
    positions: torch.Tensor,
    *,
    start_layer: int,
    flatten_pages: bool,
) -> list[dict[str, Any]]:
    records = []
    for local_layer, buffer in enumerate(buffers):
        source = buffer.flatten(0, 1) if flatten_pages else buffer
        device_slots = slots.to(device=source.device, dtype=torch.int64)
        rows = source.index_select(0, device_slots)
        layer_id = start_layer + local_layer
        records.append(
            {
                "layer_id": layer_id,
                "row_count": int(rows.shape[0]),
                "row_nbytes": (
                    int(rows[0].numel() * rows[0].element_size())
                    if rows.shape[0]
                    else 0
                ),
                "dtype": str(rows.dtype),
                "content_xor_sha256": hash_positioned_rows(
                    rows, positions, layer_id=layer_id
                ),
            }
        )
    return records


def _emit(payload: dict[str, Any]) -> None:
    logger.info("%s%s", _LOG_PREFIX, json.dumps(payload, sort_keys=True))


def _required_int_env(name: str, *, minimum: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        raise ValueError(f"{name} must be set when {_PREFILL_LAYER_ENABLE_ENV}=1")
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from error
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _parse_sparse_layer_ids(raw_value: str) -> frozenset[int]:
    raw_items = raw_value.split(",")
    if not raw_items or any(not item.strip() for item in raw_items):
        raise ValueError(
            f"{_PREFILL_LAYER_IDS_ENV} must be a comma-separated list of integers"
        )
    try:
        layer_ids = [int(item) for item in raw_items]
    except ValueError as error:
        raise ValueError(
            f"{_PREFILL_LAYER_IDS_ENV} must be a comma-separated list of integers, got {raw_value!r}"
        ) from error
    if any(layer_id < 0 for layer_id in layer_ids):
        raise ValueError(
            f"{_PREFILL_LAYER_IDS_ENV} values must be >= 0, got {raw_value!r}"
        )
    if len(set(layer_ids)) != len(layer_ids):
        raise ValueError(
            f"{_PREFILL_LAYER_IDS_ENV} must not contain duplicates, got {raw_value!r}"
        )
    return frozenset(layer_ids)


def _sub_layer_hash_enabled() -> bool:
    raw_value = os.getenv(_PREFILL_SUB_LAYER_ENABLE_ENV, "0")
    if raw_value not in {"0", "1"}:
        raise ValueError(
            f"{_PREFILL_SUB_LAYER_ENABLE_ENV} must be 0 or 1, got {raw_value!r}"
        )
    return raw_value == "1"


def _mlp_hash_enabled(*, sub_layer_enabled: bool) -> bool:
    raw_value = os.getenv(_PREFILL_MLP_ENABLE_ENV, "0")
    if raw_value not in {"0", "1"}:
        raise ValueError(f"{_PREFILL_MLP_ENABLE_ENV} must be 0 or 1, got {raw_value!r}")
    enabled = raw_value == "1"
    if enabled and not sub_layer_enabled:
        raise ValueError(
            f"{_PREFILL_MLP_ENABLE_ENV}=1 requires {_PREFILL_SUB_LAYER_ENABLE_ENV}=1"
        )
    return enabled


def _mlp_all_rank_hash_enabled(*, mlp_enabled: bool) -> bool:
    raw_value = os.getenv(_PREFILL_MLP_ALL_RANK_ENABLE_ENV, "0")
    if raw_value not in {"0", "1"}:
        raise ValueError(
            f"{_PREFILL_MLP_ALL_RANK_ENABLE_ENV} must be 0 or 1, got {raw_value!r}"
        )
    enabled = raw_value == "1"
    if enabled and not mlp_enabled:
        raise ValueError(
            f"{_PREFILL_MLP_ALL_RANK_ENABLE_ENV}=1 requires {_PREFILL_MLP_ENABLE_ENV}=1"
        )
    return enabled


def _mlp_all_rank_internal_hash_enabled(*, all_rank_outer_reduce_enabled: bool) -> bool:
    raw_value = os.getenv(_PREFILL_MLP_ALL_RANK_INTERNAL_ENABLE_ENV, "0")
    if raw_value not in {"0", "1"}:
        raise ValueError(
            f"{_PREFILL_MLP_ALL_RANK_INTERNAL_ENABLE_ENV} must be 0 or 1, got {raw_value!r}"
        )
    enabled = raw_value == "1"
    if enabled and not all_rank_outer_reduce_enabled:
        raise ValueError(
            f"{_PREFILL_MLP_ALL_RANK_INTERNAL_ENABLE_ENV}=1 requires {_PREFILL_MLP_ALL_RANK_ENABLE_ENV}=1"
        )
    return enabled


def get_prefill_layer_hash_config(
    *,
    batch_size: int,
    is_extend: bool,
    extend_seq_lens: Sequence[int] | None,
) -> PrefillLayerHashConfig | None:
    """Return the opt-in layer-hash plan for one exact Prefill request."""
    if os.getenv(_PREFILL_LAYER_ENABLE_ENV, "0") != "1":
        return None

    target_seq_len = _required_int_env(_PREFILL_LAYER_SEQ_LEN_ENV, minimum=1)
    raw_layer_ids = os.getenv(_PREFILL_LAYER_IDS_ENV)
    has_range = (
        os.getenv(_PREFILL_LAYER_MIN_ENV) is not None
        or os.getenv(_PREFILL_LAYER_MAX_ENV) is not None
    )
    if raw_layer_ids is not None and has_range:
        raise ValueError(
            f"{_PREFILL_LAYER_IDS_ENV} and {_PREFILL_LAYER_MIN_ENV}/{_PREFILL_LAYER_MAX_ENV} are mutually exclusive"
        )
    if raw_layer_ids is not None:
        layer_ids = _parse_sparse_layer_ids(raw_layer_ids)
        min_layer = min(layer_ids)
        max_layer = max(layer_ids)
    else:
        layer_ids = None
        min_layer = _required_int_env(_PREFILL_LAYER_MIN_ENV, minimum=0)
        max_layer = _required_int_env(_PREFILL_LAYER_MAX_ENV, minimum=0)
        if min_layer > max_layer:
            raise ValueError(
                f"invalid layer range: min_layer={min_layer}, max_layer={max_layer}"
            )

    if not is_extend or batch_size != 1 or extend_seq_lens is None:
        return None
    if len(extend_seq_lens) != 1 or int(extend_seq_lens[0]) != target_seq_len:
        return None
    log_sub_layer_boundaries = _sub_layer_hash_enabled()
    log_mlp_boundaries = _mlp_hash_enabled(sub_layer_enabled=log_sub_layer_boundaries)
    log_all_rank_outer_reduce_boundaries = _mlp_all_rank_hash_enabled(
        mlp_enabled=log_mlp_boundaries
    )
    log_all_rank_mlp_internal_boundaries = _mlp_all_rank_internal_hash_enabled(
        all_rank_outer_reduce_enabled=log_all_rank_outer_reduce_boundaries
    )
    is_primary_tp_rank = get_parallel().tp_rank == 0
    if not is_primary_tp_rank and not log_all_rank_outer_reduce_boundaries:
        return None
    return PrefillLayerHashConfig(
        seq_len=target_seq_len,
        min_layer=min_layer,
        max_layer=max_layer,
        layer_ids=layer_ids,
        log_layer_boundaries=is_primary_tp_rank,
        log_sub_layer_boundaries=(is_primary_tp_rank and log_sub_layer_boundaries),
        log_mlp_boundaries=is_primary_tp_rank and log_mlp_boundaries,
        log_all_rank_outer_reduce_boundaries=(log_all_rank_outer_reduce_boundaries),
        log_all_rank_mlp_internal_boundaries=(log_all_rank_mlp_internal_boundaries),
    )


def log_prefill_layer_hash_snapshot(
    *,
    boundary: str,
    layer_id: int,
    rid: str,
    seq_len: int,
    positions: torch.Tensor,
    tensors: dict[str, torch.Tensor | None],
) -> None:
    """Log content-free hashes for one Prefill layer boundary."""
    if positions.ndim != 1 or positions.numel() < seq_len:
        raise ValueError(
            f"prefill layer hash needs at least {seq_len} positions, got {positions.shape}"
        )
    # MLP-sync appends alignment rows after the request tokens. Compare only
    # the logical sequence so the diagnostic does not hash padding state.
    positions = positions[:seq_len]

    common = {
        "record_type": "prefill_layer_tensor_hash",
        "boundary": boundary,
        "layer_id": int(layer_id),
        "rid_sha256": hashlib.sha256(str(rid).encode()).hexdigest(),
        "seq_len": int(seq_len),
        "positions_sha256": hash_int_sequence(positions),
        "topology": _topology(),
    }
    for component, tensor in sorted(tensors.items()):
        if tensor is None:
            _emit(
                {
                    **common,
                    "component": component,
                    "present": False,
                    "shape": None,
                    "dtype": None,
                    "row_count": 0,
                    "row_nbytes": 0,
                    "content_xor_sha256": None,
                }
            )
            continue
        if tensor.ndim == 0 or tensor.shape[0] < seq_len:
            raise ValueError(
                f"prefill layer hash component {component!r} needs at least {seq_len} rows, got {tensor.shape}"
            )
        tensor = tensor[:seq_len]
        _emit(
            {
                **common,
                "component": component,
                "present": True,
                "shape": [int(size) for size in tensor.shape],
                "dtype": str(tensor.dtype),
                "row_count": int(tensor.shape[0]),
                "row_nbytes": int(tensor[0].numel() * tensor[0].element_size()),
                "content_xor_sha256": hash_positioned_rows(
                    tensor,
                    positions,
                    layer_id=layer_id,
                ),
            }
        )


def log_dsa_cache_hash_snapshot(
    *,
    stage: str,
    rid: str,
    bootstrap_room: int | None,
    seq_len: int,
    global_slots: torch.Tensor,
    pool: Any,
) -> bool:
    """Log main latent-KV and replicated DSA index-cache hashes once enabled."""
    if not should_log_cache_hash(seq_len):
        return False
    if global_slots.numel() < seq_len:
        raise ValueError(
            f"cache hash needs {seq_len} slots, got {global_slots.numel()}"
        )
    kv_buffers = getattr(pool, "kv_buffer", None)
    index_buffers = getattr(pool, "index_k_buffer", None)
    if not kv_buffers or not index_buffers:
        raise RuntimeError(
            "DCP cache hash diagnostic requires a DSA pool with latent KV and plain index_k_buffer storage"
        )

    slots = global_slots[:seq_len].to(dtype=torch.int64)
    topology = _topology()
    positions, local_slots = build_owned_slot_plan(
        slots,
        dcp_size=topology["dcp_size"],
        dcp_rank=topology["dcp_rank"],
    )
    owned_global_slots = slots.index_select(0, positions.to(device=slots.device))
    start_layer = int(getattr(pool, "start_layer", 0) or 0)
    common = {
        "stage": stage,
        "rid_sha256": hashlib.sha256(str(rid).encode()).hexdigest(),
        "bootstrap_room": int(bootstrap_room or 0),
        "seq_len": int(seq_len),
        "topology": topology,
    }
    _emit(
        {
            **common,
            "record_type": "slot_map",
            "owned_position_count": int(positions.numel()),
            "owned_positions_sha256": hash_int_sequence(positions),
            "owned_global_slots_sha256": hash_int_sequence(owned_global_slots),
            "local_slots_sha256": hash_int_sequence(local_slots),
        }
    )

    components = (
        (
            "latent_kv",
            _layer_records(
                kv_buffers,
                local_slots,
                positions,
                start_layer=start_layer,
                flatten_pages=False,
            ),
        ),
        (
            "dsa_index_k",
            _layer_records(
                index_buffers,
                slots,
                torch.arange(seq_len, dtype=torch.int64, device=slots.device),
                start_layer=start_layer,
                flatten_pages=True,
            ),
        ),
    )
    for component, records in components:
        for record in records:
            _emit(
                {
                    **common,
                    "record_type": "layer_hash",
                    "component": component,
                    **record,
                }
            )
    return True


def log_dcp_transfer_plan(
    *,
    session_id: str,
    seq_len: int,
    physical_page_size: int,
    dcp_size: int,
    dcp_rank: int,
    src_page_offset: int,
    decode_prefix_len: int,
    src_page_indices: np.ndarray,
    dst_page_indices: np.ndarray,
    src_token_indices: np.ndarray,
    dst_token_indices: np.ndarray,
) -> bool:
    if not should_log_cache_hash(seq_len):
        return False
    _emit(
        {
            "record_type": "transfer_plan",
            "session_sha256": hashlib.sha256(session_id.encode()).hexdigest(),
            "seq_len": int(seq_len),
            "physical_page_size": int(physical_page_size),
            "dcp_size": int(dcp_size),
            "dcp_rank": int(dcp_rank),
            "src_page_offset": int(src_page_offset),
            "decode_prefix_len": int(decode_prefix_len),
            "src_pages_count": int(len(src_page_indices)),
            "dst_pages_count": int(len(dst_page_indices)),
            "token_count": int(len(src_token_indices)),
            "src_pages_sha256": hash_int_sequence(src_page_indices),
            "dst_pages_sha256": hash_int_sequence(dst_page_indices),
            "src_token_indices_sha256": hash_int_sequence(src_token_indices),
            "dst_token_indices_sha256": hash_int_sequence(dst_token_indices),
        }
    )
    return True
