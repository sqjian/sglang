# to be combined with the sparse coordinator class and sparse algorithm family

import logging
from typing import List, NamedTuple, Union

import torch
import torch.cuda.nvtx as nvtx

from sglang.jit_kernel.hisparse import (
    execute_h2d_copy_mla,
    finalize_accepted,
    prepare_swap_mla,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.hisparse_memory_pool import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    DeepSeekV4SingleKVPoolHost,
    HiSparseNSATokenToKVPool,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.memory_pool_host import MLATokenToKVPoolHost
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.utils import get_device_module

device_module = get_device_module()

logger = logging.getLogger(__name__)


class SwapResult(NamedTuple):
    top_k_device_locs: torch.Tensor
    hit_device_locs: torch.Tensor
    miss_device_locs: torch.Tensor
    hit_count: torch.Tensor
    miss_src_locs: torch.Tensor
    miss_dst_locs: torch.Tensor


class HiSparseAct(NamedTuple):
    start_event: device_module.Event
    finish_event: device_module.Event
    req: Req


class HiSparseTokenStats(NamedTuple):
    device_tokens: int
    device_token_usage: float
    host_tokens: int
    host_token_usage: float


class HiSparseCoordinator:
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: Union[
            HiSparseTokenToKVPoolAllocator,
            DeepSeekV4HiSparseTokenToKVPoolAllocator,
        ],
        top_k: int,
        device_buffer_size: int,
        device: str,
        tp_group,
        host_to_device_ratio: int = 2,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.top_k = top_k
        self.device_buffer_size = device_buffer_size
        self.device = device
        self.compress_ratio = self.token_to_kv_pool_allocator.compress_ratio

        self.is_dsv4_hisparse = isinstance(
            self.token_to_kv_pool_allocator, DeepSeekV4HiSparseTokenToKVPoolAllocator
        )
        if self.is_dsv4_hisparse:
            self.mem_pool_device = self.token_to_kv_pool_allocator.hisparse_kvcache
            host_size = self.token_to_kv_pool_allocator.size_full // self.compress_ratio
            self.mem_pool_host = DeepSeekV4SingleKVPoolHost(
                self.mem_pool_device, host_size, 1
            )
            self.item_size_bytes = (
                self.mem_pool_host.kv_cache_total_dim
                * self.mem_pool_host.dtype.itemsize
            )
        else:
            assert isinstance(
                self.token_to_kv_pool_allocator, HiSparseTokenToKVPoolAllocator
            )
            self.mem_pool_device: HiSparseNSATokenToKVPool = (
                self.token_to_kv_pool_allocator.get_kvcache()
            )
            logical_host_size = self._logical_host_size()
            effective_host_to_device_ratio = max(
                host_to_device_ratio,
                logical_host_size / self.mem_pool_device.size,
            )
            self.mem_pool_host = MLATokenToKVPoolHost(
                device_pool=self.mem_pool_device,
                host_to_device_ratio=effective_host_to_device_ratio,
                host_size=0,
                page_size=1,  # enable backup one token at a time
                layout="layer_first",
                override_kv_cache_dim=self.mem_pool_device.kv_cache_dim,
            )
            self.item_size_bytes = self.mem_pool_host.token_stride_size

        max_num_req_slots = req_to_token_pool.req_to_token.shape[0]
        max_context_len = req_to_token_pool.max_context_len
        max_compressed_context_len = (
            max_context_len + self.compress_ratio - 1
        ) // self.compress_ratio

        # to have an extra page for new tokens
        self.padded_buffer_size = (
            self.device_buffer_size + self.mem_pool_device.page_size
        )

        self.req_to_device_buffer = torch.zeros(
            (max_num_req_slots, self.padded_buffer_size),
            dtype=torch.int64,
            device=device,
        )
        self.req_device_buffer_size = torch.zeros(
            max_num_req_slots, dtype=torch.int64, device="cpu"
        )
        self.req_device_buffer_regular_size = torch.zeros(
            max_num_req_slots, dtype=torch.int64, device="cpu"
        )
        self.req_device_buffer_regular_size_gpu = torch.zeros(
            max_num_req_slots, dtype=torch.int64, device=device
        )
        self.req_to_host_pool = torch.full(
            (max_num_req_slots, max_compressed_context_len),
            -1,
            dtype=torch.int64,
            device=device,
        )

        self.write_staging_stream = device_module.Stream()
        self.decode_backup_stream = device_module.Stream()
        self.ack_staging_queue: List[HiSparseAct] = []
        self.decode_producer_stream = None
        self._backup_done_event = device_module.Event()
        self._has_pending_backup = False

        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

        # initialize data structures for swap-in kernel
        layer_num = self.mem_pool_device.layer_num
        self.req_device_buffer_tokens = torch.full(
            (layer_num, max_num_req_slots, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.req_device_buffer_token_locs = torch.full(
            (layer_num, max_num_req_slots, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._lru_init = torch.arange(
            self.device_buffer_size, dtype=torch.int16, device=device
        )
        self.lru_slots = (
            self._lru_init.view(1, 1, -1)
            .repeat(layer_num, max_num_req_slots, 1)
            .contiguous()
        )
        self._device_buffer_arange_i32 = torch.arange(
            self.device_buffer_size, dtype=torch.int32, device=device
        )

        # Pre-allocated output buffer for swap_in_selected_pages (CUDA-graph safe)
        self.top_k_device_locs_buffer = torch.full(
            (max_num_req_slots, self.top_k), -1, dtype=torch.int32, device=device
        )
        self.raw_indices_buffer = torch.full(
            (max_num_req_slots, self.top_k), -1, dtype=torch.int32, device=device
        )
        # Scalar tensor: number of real (non-padded) requests in the batch.
        # Updated before each graph replay so padded blocks early-return.
        self.num_real_reqs = torch.zeros(1, dtype=torch.int32, device=device)

        # --- Dual-attention overlap buffers ---
        self.transfer_stream = device_module.Stream()
        self.h2d_start_event = device_module.Event()
        self.h2d_finish_event = device_module.Event()
        self.d2h_finish_event = device_module.Event()
        # Pre-record so the first swap_in_selected_pages() wait passes immediately
        # (finalize_accepted_tokens hasn't been called yet on the first decode step).
        self.d2h_finish_event.record()

        self.hit_device_locs_buffer = torch.full(
            (max_num_req_slots, self.top_k), -1, dtype=torch.int32, device=device
        )
        self.miss_device_locs_buffer = torch.full(
            (max_num_req_slots, self.top_k), -1, dtype=torch.int32, device=device
        )
        self.hit_count_buffer = torch.empty(
            (max_num_req_slots,), dtype=torch.int32, device=device
        )
        self.miss_src_locs_buffer = torch.empty(
            (max_num_req_slots, self.top_k), dtype=torch.int64, device=device
        )
        self.miss_dst_locs_buffer = torch.empty(
            (max_num_req_slots, self.top_k), dtype=torch.int64, device=device
        )

        # CPU flag: True means "skip backup on the next decode step" because
        # staging already backed up all prefill tokens.  Cleared after one step.
        self._skip_first_backup = [False] * max_num_req_slots
        self._exact_host_slots = [False] * max_num_req_slots

        # Buffers for finalize_accepted_tokens kernel
        self.needs_move_buffer = torch.zeros(
            max_num_req_slots, dtype=torch.int32, device=device
        )
        self.last_accepted_device_buffer = torch.empty(
            max_num_req_slots, dtype=torch.int64, device=device
        )
        self.newest_slot_device_buffer = torch.empty(
            max_num_req_slots, dtype=torch.int64, device=device
        )
        self.cumsum_buffer = torch.empty(
            max_num_req_slots + 1, dtype=torch.int64, device=device
        )

    def _logical_host_size(self) -> int:
        logical_size = getattr(self.token_to_kv_pool_allocator, "size_full", None)
        if logical_size is None:
            logical_allocator = getattr(
                self.token_to_kv_pool_allocator, "logical_attn_allocator", None
            )
            logical_size = getattr(logical_allocator, "size", self.mem_pool_device.size)
        page_size = getattr(self.token_to_kv_pool_allocator, "page_size", 1)
        return int(logical_size) + int(page_size)

    def host_pool_debug_info(self, indices: torch.Tensor = None) -> str:
        logical_max = self._logical_host_size() - 1
        parts = [
            f"host_available={self.mem_pool_host.available_size()}",
            f"host_size={self.mem_pool_host.size}",
            f"logical_max={logical_max}",
        ]
        if indices is not None and indices.numel() > 0:
            indices_cpu = indices.detach().to(dtype=torch.int64, device="cpu")
            parts.extend(
                [
                    f"requested_min={int(indices_cpu.min())}",
                    f"requested_max={int(indices_cpu.max())}",
                    f"requested_count={indices_cpu.numel()}",
                ]
            )
        return ", ".join(parts)

    def _alloc_exact_host_or_existing(
        self,
        req_idx: int,
        token_positions: torch.Tensor,
        logical_locs: torch.Tensor,
        context: str,
    ) -> torch.Tensor:
        token_positions_cpu = token_positions.detach().to(
            dtype=torch.int64, device="cpu"
        )
        logical_locs_cpu = logical_locs.detach().to(dtype=torch.int64, device="cpu")
        existing = (
            self.req_to_host_pool[
                req_idx, token_positions_cpu.to(device=self.req_to_host_pool.device)
            ]
            .detach()
            .to(dtype=torch.int64, device="cpu")
        )

        invalid_existing = (existing >= 0) & (existing != logical_locs_cpu)
        if torch.any(invalid_existing).item():
            invalid_idx = torch.where(invalid_existing)[0][0]
            bad_pos = int(token_positions_cpu[invalid_idx])
            raise RuntimeError(
                "HiSparse exact host slot conflict in "
                f"{context}: req_idx={req_idx}, token_pos={bad_pos}, "
                f"existing_host={int(existing[invalid_idx])}, "
                f"logical_loc={int(logical_locs_cpu[invalid_idx])}; "
                f"{self.host_pool_debug_info(logical_locs_cpu)}"
            )

        missing = existing < 0
        if torch.any(missing).item():
            missing_locs = logical_locs_cpu[missing]
            allocated = self.mem_pool_host.alloc_specific(missing_locs)
            if allocated is None:
                missing_pos = token_positions_cpu[missing]
                raise RuntimeError(
                    "HiSparse exact host slot allocation failed in "
                    f"{context}: req_idx={req_idx}, "
                    f"token_pos_min={int(missing_pos.min())}, "
                    f"token_pos_max={int(missing_pos.max())}; "
                    f"{self.host_pool_debug_info(missing_locs)}"
                )

        return logical_locs_cpu

    def set_decode_producer_stream(self, stream) -> None:
        self.decode_producer_stream = stream

    def get_token_stats(self) -> HiSparseTokenStats:
        device_allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        device_capacity = device_allocator.size
        device_tokens = device_capacity - device_allocator.available_size()
        host_capacity = self.mem_pool_host.size
        host_tokens = host_capacity - self.mem_pool_host.available_size()
        return HiSparseTokenStats(
            device_tokens=device_tokens,
            device_token_usage=(
                device_tokens / device_capacity if device_capacity > 0 else 0.0
            ),
            host_tokens=host_tokens,
            host_token_usage=(
                host_tokens / host_capacity if host_capacity > 0 else 0.0
            ),
        )

    def admit_request_into_staging(self, req: Req) -> None:
        req.hisparse_staging = True

        full_kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(req.fill_ids)
        ].to(dtype=torch.int64, copy=True)
        device_indices = (
            self.mem_pool_device.translate_loc_from_full_to_hisparse_device(
                full_kv_indices
            )
        )

        prefill_len = len(device_indices)
        host_indices = self.mem_pool_host.alloc(prefill_len)
        if host_indices is None:
            logger.error(
                "HiSparse: host mem pool alloc failed for %d tokens (req %s)",
                prefill_len,
                req.rid,
            )
            raise RuntimeError(
                f"HiSparse host mem pool alloc failed for {prefill_len} tokens"
            )
        host_indices = host_indices.to(device=self.device)
        self.req_to_host_pool[req.req_pool_idx, :prefill_len] = host_indices

        start_event = device_module.Event()
        finish_event = device_module.Event()
        start_event.record()
        with device_module.stream(self.write_staging_stream):
            start_event.wait(self.write_staging_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_indices,
                device_indices,
                io_backend="kernel",
            )
            finish_event.record()
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_staging_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_staging_stream)

        self.ack_staging_queue.append(HiSparseAct(start_event, finish_event, req))

    def admit_request_direct(
        self, req: Req, require_spec_extra_page: bool = False
    ) -> None:
        if not self.try_admit_request_direct(req, require_spec_extra_page):
            alloc_size, regular_size = self.estimate_device_buffer_alloc_size(
                req, require_spec_extra_page
            )
            available = (
                self.token_to_kv_pool_allocator.hisparse_attn_allocator.available_size()
            )
            raise RuntimeError(
                "HiSparse direct admit failed: insufficient device buffer capacity "
                f"for req {req.rid} (alloc_size={alloc_size}, "
                f"regular_size={regular_size}, available={available})"
            )

    def can_admit_request_direct(
        self, req: Req, require_spec_extra_page: bool = False
    ) -> bool:
        alloc_size, _ = self.estimate_device_buffer_alloc_size(
            req, require_spec_extra_page
        )
        allocated_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : req.kv_allocated_len
        ]
        extra_size = self._estimate_device_buffer_extra_alloc_size(
            allocated_indices, alloc_size
        )
        available = (
            self.token_to_kv_pool_allocator.hisparse_attn_allocator.available_size()
        )
        return extra_size <= available

    def try_admit_request_direct(
        self, req: Req, require_spec_extra_page: bool = False
    ) -> bool:
        """Direct-to-host path: KV data already resides in host pool via RDMA.

        Skips staging DMA entirely. Only allocates a small device buffer
        (4KB) for decode-time swap-in, then marks the request as ready.
        Host indices were already written to req_to_host_pool.

        Metadata fixups after alloc_device_buffer():
        - alloc_device_buffer() sets device_buffer_tokens = [0, 1, ..., buf_size-1],
          which tells the swap-in kernel that those tokens are cached in the device
          buffer.  In the staging path this is correct (prefill filled the buffer),
          but here the buffer is empty.
        """
        if self.is_dsv4_hisparse:
            # TODO(dsv4): wire PD direct-to-host. Needs (a) load_to_device_per_layer
            raise NotImplementedError(
                "PD direct-to-host admission is not supported for dsv4 hisparse yet."
            )

        if not self.can_admit_request_direct(req, require_spec_extra_page):
            return False
        if not self.alloc_device_buffer(req, require_spec_extra_page):
            return False

        if req.kv_allocated_len <= self.device_buffer_size:
            # Short sequences (seq_len <= device_buffer_size): the kernel fast path
            # returns device_buffer_locs directly without any host loading, so we
            # must preload all tokens from host pool into the device buffer
            # TODO(hzh0425): Optimize this.
            self._preload_to_device_buffer(req)
        else:
            # Long sequence: reset device_buffer_tokens to -1 so the kernel
            # sees all slots as empty -> every top-k lookup is a miss -> host load.
            self.req_device_buffer_tokens[
                :, req.req_pool_idx, : self.device_buffer_size
            ] = -1

        req.hisparse_staging = False
        self._exact_host_slots[req.req_pool_idx] = bool(
            getattr(req, "hisparse_exact_host_slots", False)
        )
        self._skip_first_backup[req.req_pool_idx] = True
        logger.debug("HiSparse: admitting request %s directly", req.rid)
        return True

    def _preload_to_device_buffer(self, req: Req) -> None:
        """Preload all tokens from host pool into the device buffer."""
        n = req.kv_allocated_len
        host_indices = self.req_to_host_pool[req.req_pool_idx, :n]
        device_locs = self.req_to_device_buffer[req.req_pool_idx, :n]

        for layer_id in range(self.mem_pool_device.layer_num):
            self.mem_pool_host.load_to_device_per_layer(
                self.mem_pool_device,
                host_indices,
                device_locs,
                layer_id,
                io_backend="kernel",
            )

    def alloc_device_buffer(
        self, req: Req, require_spec_extra_page: bool = False
    ) -> bool:
        if self.is_dsv4_hisparse:
            allocated_len = len(req.fill_ids)
            alloc_size = self.padded_buffer_size
            regular_size = self.device_buffer_size
        else:
            allocated_len = req.kv_allocated_len
            alloc_size, regular_size = self.estimate_device_buffer_alloc_size(
                req, require_spec_extra_page
            )

        allocated_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :allocated_len
        ]
        compressed_logical_indices = (
            self.mem_pool_device.translate_loc_from_full_to_compressed(allocated_indices)
        )
        compressed_len = len(compressed_logical_indices)
        buffer_indices = self.token_to_kv_pool_allocator.alloc_device_buffer(
            compressed_logical_indices, alloc_size
        )
        if buffer_indices is None:
            logger.debug(
                "HiSparse: alloc_device_buffer failed for req %s "
                "(compressed_len=%d, alloc_size=%d)",
                req.rid,
                compressed_len,
                alloc_size,
            )
            return False

        buffer_indices = buffer_indices.to(torch.int32)
        self.req_to_device_buffer[req.req_pool_idx, :alloc_size] = buffer_indices
        self.req_device_buffer_size[req.req_pool_idx] = alloc_size
        self.req_device_buffer_regular_size[req.req_pool_idx] = regular_size
        self.req_device_buffer_regular_size_gpu[req.req_pool_idx] = regular_size

        self.req_device_buffer_tokens[
            :, req.req_pool_idx, : self.device_buffer_size
        ] = self._device_buffer_arange_i32
        self.req_device_buffer_token_locs[:, req.req_pool_idx, :alloc_size] = (
            buffer_indices[:alloc_size]
        )

        if not self.is_dsv4_hisparse and req.kv_allocated_len <= regular_size:
            # EAGLE target-verify runs through the extend attention path, which
            # translates req_to_token logical locs directly instead of using the
            # decode swap-in coordinator. Short prompts are fully resident in the
            # device buffer, so keep their logical->device mapping visible.
            self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping[
                allocated_indices
            ] = buffer_indices[: req.kv_allocated_len]
        return True

    def estimate_device_buffer_alloc_size(
        self, req: Req, require_spec_extra_page: bool = False
    ) -> tuple[int, int]:
        return self.estimate_device_buffer_alloc_size_for_len(
            req.kv_allocated_len,
            req.sampling_params.max_new_tokens,
            require_spec_extra_page,
        )

    def estimate_device_buffer_alloc_size_for_len(
        self,
        kv_allocated_len: int,
        max_new_tokens: int,
        require_spec_extra_page: bool = False,
    ) -> tuple[int, int]:
        page_size = self.mem_pool_device.page_size
        reserve_tokens = max_new_tokens if require_spec_extra_page else 0
        regular_size = min(
            ((kv_allocated_len + reserve_tokens + page_size - 1) // page_size)
            * page_size,
            self.device_buffer_size,
        )
        alloc_size = regular_size
        if regular_size == self.device_buffer_size:
            alloc_size = self.padded_buffer_size
        elif require_spec_extra_page:
            alloc_size = regular_size + page_size
        return alloc_size, regular_size

    def _estimate_device_buffer_extra_alloc_size(
        self, allocated_indices: torch.Tensor, alloc_size: int
    ) -> int:
        hisparse_indices = (
            self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping[
                allocated_indices
            ]
        )
        mapped_size = int(torch.count_nonzero(hisparse_indices > 0).item())
        if mapped_size >= alloc_size:
            return 0

        page_size = self.mem_pool_device.page_size
        page_residual_size = mapped_size % page_size
        if page_residual_size != 0:
            mapped_size += page_size - page_residual_size
        return max(alloc_size - mapped_size, 0)

    def has_ongoing_staging(self) -> bool:
        return len(self.ack_staging_queue) > 0

    def collect_ready_reqs(self) -> List[Req]:
        ready_reqs = []
        if len(self.ack_staging_queue) == 0:
            return ready_reqs

        finish_count = 0
        for _, finish_event, _ in self.ack_staging_queue:
            if not finish_event.query():
                break
            finish_count += 1
        queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        if self.tp_world_size > 1:
            # synchronize TP workers to make sure the same update to scheduler
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
        finish_count = int(queue_size.item())
        while finish_count > 0:
            _, _, req = self.ack_staging_queue.pop(0)
            # prepare device buffer and update req
            require_spec_extra_page = (
                getattr(req, "hisparse_spec_info", None) is not None
            )
            if not self.alloc_device_buffer(req, require_spec_extra_page):
                alloc_size, regular_size = self.estimate_device_buffer_alloc_size(
                    req, require_spec_extra_page
                )
                available = (
                    self.token_to_kv_pool_allocator.hisparse_attn_allocator.available_size()
                )
                raise RuntimeError(
                    "HiSparse staging admit failed: insufficient device buffer "
                    f"capacity for req {req.rid} (alloc_size={alloc_size}, "
                    f"regular_size={regular_size}, available={available})"
                )
            req.hisparse_staging = False
            self._skip_first_backup[req.req_pool_idx] = True
            finish_count -= 1
            ready_reqs.append(req)
        return ready_reqs

    def _grow_device_buffers(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """Grow device buffers for requests whose sequence length exceeds current capacity."""
        current_caps = self.req_device_buffer_size[req_pool_indices_cpu]
        short_reqs_cpu = seq_lens_cpu <= self.device_buffer_size
        needs_grow_cpu = short_reqs_cpu & (seq_lens_cpu > current_caps)

        if torch.any(needs_grow_cpu):
            page_size = self.mem_pool_device.page_size
            grow_indices = torch.where(needs_grow_cpu)[0]

            # Compute all grow sizes on CPU, then do a single bulk allocation
            req_idxs = []
            old_caps = []
            new_caps = []
            grow_sizes = []
            total_grow = 0
            for i in grow_indices.tolist():
                req_idx = int(req_pool_indices_cpu[i])
                current_cap = int(current_caps[i])
                seq_len = int(seq_lens_cpu[i])

                new_cap = min(
                    ((seq_len + page_size - 1) // page_size) * page_size,
                    self.device_buffer_size,
                )
                if new_cap == self.device_buffer_size:
                    new_cap = self.padded_buffer_size
                grow_size = new_cap - current_cap
                if grow_size <= 0:
                    continue
                req_idxs.append(req_idx)
                old_caps.append(current_cap)
                new_caps.append(new_cap)
                grow_sizes.append(grow_size)
                total_grow += grow_size

            if total_grow > 0:
                all_new_indices = (
                    self.token_to_kv_pool_allocator.hisparse_attn_allocator.alloc(
                        total_grow
                    )
                )
                if all_new_indices is None:
                    logger.error(
                        "HiSparse: _grow_device_buffers bulk alloc failed "
                        "(total_grow=%d)",
                        total_grow,
                    )
                    raise RuntimeError(
                        f"HiSparse _grow_device_buffers failed (total_grow={total_grow})"
                    )

                offset = 0
                for req_idx, current_cap, new_cap, grow_size in zip(
                    req_idxs, old_caps, new_caps, grow_sizes
                ):
                    chunk = all_new_indices[offset : offset + grow_size]
                    offset += grow_size
                    self.req_to_device_buffer[req_idx, current_cap:new_cap] = chunk
                    self.req_device_buffer_token_locs[
                        :, req_idx, current_cap:new_cap
                    ] = chunk
                    self.req_device_buffer_size[req_idx] = new_cap

        reserved_positions = (seq_lens - 1).clamp(max=self.device_buffer_size)
        return self.req_to_device_buffer[req_pool_indices, reserved_positions]

    def has_ongoing_staging(self) -> bool:
        return len(self.ack_staging_queue) > 0

    def collect_ready_reqs(self) -> List[Req]:
        ready_reqs: List[Req] = []
        if len(self.ack_staging_queue) == 0:
            return ready_reqs

        finish_count = 0
        for _, finish_event, _ in self.ack_staging_queue:
            if not finish_event.query():
                break
            finish_count += 1
        queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        if self.tp_world_size > 1:
            # synchronize TP workers to make sure the same update to scheduler
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
        finish_count = int(queue_size.item())
        while finish_count > 0:
            _, _, req = self.ack_staging_queue.pop(0)
            # prepare device buffer and update req
            self.alloc_device_buffer(req)
            self._skip_first_backup[req.req_pool_idx] = True
            req.hisparse_staging = False
            finish_count -= 1
            ready_reqs.append(req)
        return ready_reqs

    def map_last_loc_to_buffer(
        self,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
    ) -> None:
        req_pool_indices_cpu = req_pool_indices.cpu()

        self._eager_backup_previous_token(
            seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
        )

        if not self.is_dsv4_hisparse:
            # Grow device buffers if needed and resolve the latest-token slot.
            reserved_buffer_loc = self._grow_device_buffers(
                seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
            )
            self.req_device_buffer_token_locs[
                :, req_pool_indices, self.device_buffer_size
            ] = reserved_buffer_loc.to(torch.int32)

            # No need to clear prior mappings: the only consumer of the mapping
            # for past tokens is the swap-in kernel, and it goes through
            # top_k_device_locs returned by swap_in_selected_pages -- not via
            # mapping[old_out_cache_loc] -- so stale entries are harmless.
            compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(
                out_cache_loc
            )
            self.mem_pool_device.full_to_hisparse_device_index_mapping[
                compressed_locs
            ] = reserved_buffer_loc
            return

        active_reqs = seq_lens % self.compress_ratio == 0
        if not torch.any(active_reqs):
            return

        active_seq_lens = seq_lens[active_reqs]
        active_out_cache_loc = out_cache_loc[active_reqs]
        active_req_pool_indices = req_pool_indices[active_reqs]

        compressed_seq_lens = active_seq_lens // self.compress_ratio
        reserved_positions = (compressed_seq_lens - 1).clamp(
            max=self.device_buffer_size
        )
        reserved_buffer_loc = self.req_to_device_buffer[
            active_req_pool_indices, reserved_positions
        ]

        self.req_device_buffer_token_locs[
            :, active_req_pool_indices, self.device_buffer_size
        ] = reserved_buffer_loc.to(torch.int32)

        compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(
            active_out_cache_loc
        )
        self.mem_pool_device.full_to_hisparse_device_index_mapping[compressed_locs] = (
            reserved_buffer_loc
        )

    def _eager_backup_previous_token(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        """Back up the previous compressed token to host memory.

        Each newly produced compressed token (one per `compress_ratio` decode
        steps) must be backed up to host so the swap-in kernel can later
        recover it.

        Two cases are skipped:
        - The first decode step right after staging: all prefill tokens were
          already backed up during staging, so there is nothing new to save.
        - Steps where `(seq_len - 1) % compress_ratio != 0`: no new compressed
          token was produced this step.
        """
        # Build the list of batch positions that need a host backup.
        # Skip the first decode step after staging (prefill already backed up),
        # and skip non-aligned steps that did not produce a new compressed token.
        backup_indices = []
        for i in range(len(seq_lens_cpu)):
            req_idx = int(req_pool_indices_cpu[i])
            if self._skip_first_backup[req_idx]:
                self._skip_first_backup[req_idx] = False
                continue
            if (int(seq_lens_cpu[i]) - 1) % self.compress_ratio == 0:
                backup_indices.append(i)

        if not backup_indices:
            return

        backup_indices_gpu = torch.tensor(
            backup_indices, dtype=torch.int64, device=self.device
        )
        backup_req_indices = req_pool_indices[backup_indices_gpu]

        # The previous compressed token's position and its device buffer slot:
        #  compressed_pos = (seq_len - 1) // compress_ratio - 1
        #  - short: slot = compressed_pos          (within the regular buffer)
        #  - long:  slot = device_buffer_size      (the reserved slot)
        prev_seq_lens = seq_lens[backup_indices_gpu] - 1
        compressed_prev_seq_lens = prev_seq_lens // self.compress_ratio
        actual_compressed_pos = compressed_prev_seq_lens - 1

        buffer_slot = actual_compressed_pos.clamp(max=self.device_buffer_size)

        device_locs = self.req_to_device_buffer[backup_req_indices, buffer_slot]

        host_locs = []
        for i in range(len(backup_indices)):
            req_idx = int(backup_req_indices[i])
            token_pos = actual_token_pos[i : i + 1]
            if self._exact_host_slots[req_idx]:
                logical_loc = self.req_to_token_pool.req_to_token[
                    req_idx, token_pos
                ].reshape(-1)
                host_locs.append(
                    self._alloc_exact_host_or_existing(
                        req_idx,
                        token_pos.reshape(-1),
                        logical_loc,
                        "decode backup",
                    )
                )
            else:
                allocated = self.mem_pool_host.alloc(1)
                if allocated is None:
                    logger.error(
                        "HiSparse: host mem pool alloc failed for decode backup token"
                    )
                    raise RuntimeError(
                        "HiSparse host mem pool alloc failed for decode backup "
                        f"token; {self.host_pool_debug_info()}"
                    )
                host_locs.append(allocated)
        host_locs = torch.cat(host_locs).to(device=self.device)
        if host_locs.numel() != device_locs.numel():
            raise RuntimeError(
                "HiSparse decode backup produced mismatched host locations: "
                f"host={host_locs.numel()}, device={device_locs.numel()}"
            )
        self.req_to_host_pool[backup_req_indices, actual_compressed_pos] = host_locs

        self.wait_for_pending_backup()
        schedule_stream = device_module.current_stream()
        with device_module.stream(self.decode_backup_stream):
            self.decode_backup_stream.wait_stream(schedule_stream)
            if self.decode_producer_stream is not None:
                self.decode_backup_stream.wait_stream(self.decode_producer_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_locs,
                device_locs,
                io_backend="kernel",
            )
            self._backup_done_event.record()
            if host_locs.is_cuda:
                host_locs.record_stream(self.decode_backup_stream)
            if backup_req_indices.is_cuda:
                backup_req_indices.record_stream(self.decode_backup_stream)
            if actual_compressed_pos.is_cuda:
                actual_compressed_pos.record_stream(self.decode_backup_stream)
            if device_locs.is_cuda:
                device_locs.record_stream(self.decode_backup_stream)
        self._has_pending_backup = True

    def wait_for_pending_backup(self) -> None:
        if not self._has_pending_backup:
            return
        self._backup_done_event.wait(device_module.current_stream())
        self._has_pending_backup = False

    def naive_load_topk(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k_tokens: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Load top-k selected tokens into device memory and return their device indices.

        This is a naive per-request loop implementation for debugging/validation.
        Production code uses swap_in_selected_pages (JIT CUDA kernel) instead.

        Note: dsv4 hisparse is not supported — DeepSeekV4SingleKVPoolHost has no
        load_to_device_per_layer and indices live in compressed space. Currently
        only used as a kernel oracle in test_hisparse_unit.py (non-dsv4 path).

        Args:
            req_pool_indices: Pool indices for each request.  Shape: (num_reqs,)
            seq_lens: Sequence lengths for each request.  Shape: (num_reqs,)
            top_k_tokens: Selected token positions per request.  Shape: (num_reqs, top_k)
            layer_id: The layer to load KV cache for.

        Returns:
            Device KV cache indices for the selected tokens.  Shape: (num_reqs, top_k)
        """
        assert (
            not self.is_dsv4_hisparse
        ), "naive_load_topk is not implemented for dsv4 hisparse"
        num_reqs = req_pool_indices.size(0)
        top_k_indices = torch.full(
            (num_reqs, self.top_k), -1, dtype=torch.int32, device=self.device
        )

        for i in range(num_reqs):
            seq_len = int(seq_lens[i].item())
            top_n = min(seq_len, self.top_k)
            if top_n == 0:
                continue

            req_idx = int(req_pool_indices[i].item())
            selected_tokens = top_k_tokens[i, :top_n].to(dtype=torch.int64)

            assert torch.all(
                selected_tokens >= 0
            ), f"Req {req_idx}: selected tokens contain negative positions"
            assert torch.all(selected_tokens < seq_len), (
                f"Req {req_idx}: selected tokens {selected_tokens.tolist()} "
                f"out of range for seq_len={seq_len}"
            )

            if seq_len <= self.device_buffer_size:
                device_indices = self.req_to_device_buffer[req_idx, selected_tokens]
            else:
                device_indices = torch.empty(
                    top_n, dtype=torch.int64, device=self.device
                )

                is_latest_token = selected_tokens == (seq_len - 1)
                needs_host_load = ~is_latest_token

                device_indices[is_latest_token] = self.req_to_device_buffer[
                    req_idx, self.device_buffer_size
                ]

                num_to_load = int(needs_host_load.sum().item())
                if num_to_load > 0:
                    tokens_to_load = selected_tokens[needs_host_load]
                    host_locs = self.req_to_host_pool[req_idx, tokens_to_load]

                    invalid_mask = host_locs < 0
                    if torch.any(invalid_mask):
                        bad_positions = tokens_to_load[invalid_mask].tolist()
                        raise AssertionError(
                            f"Req {req_idx} (seq_len={seq_len}, layer={layer_id}): "
                            f"missing host backup at token positions {bad_positions}"
                        )

                    buffer_locs = self.req_to_device_buffer[req_idx, :num_to_load]
                    device_indices[needs_host_load] = buffer_locs

                    self.mem_pool_host.load_to_device_per_layer(
                        self.mem_pool_device,
                        host_locs,
                        buffer_locs,
                        layer_id,
                        io_backend="kernel",
                    )

            top_k_indices[i, :top_n] = device_indices.to(torch.int32)

        return top_k_indices

    def abort_staging_request(self, req: Req) -> None:
        """Remove a request from the staging queue and free its host + device resources.

        Must be called when aborting a request that has been admitted into staging
        but has not yet completed (i.e. req.hisparse_staging is True).
        """
        # Remove from staging queue
        self.ack_staging_queue = [
            act for act in self.ack_staging_queue if act.req is not req
        ]
        # Wait for any in-flight staging DMA to complete before freeing
        self.write_staging_stream.synchronize()

        prefill_len = len(req.fill_ids)
        allocated_locs = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :prefill_len
        ]
        self.token_to_kv_pool_allocator.free_hisparse(allocated_locs)

        # Free host memory that was allocated during admit_request_into_staging
        compressed_len = prefill_len // self.compress_ratio
        host_indices = self.req_to_host_pool[req.req_pool_idx, :compressed_len]
        host_indices = host_indices[host_indices >= 0]
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)
        self.req_to_host_pool[req.req_pool_idx, :] = -1
        self._skip_first_backup[req.req_pool_idx] = False
        self._exact_host_slots[req.req_pool_idx] = False
        req.hisparse_staging = False

    def retract_req(self, req: Req) -> None:
        if req.hisparse_staging:
            self.abort_staging_request(req)
        else:
            self.request_finished(req)

    def request_finished(self, req: Req):
        # release resources only after the execution of a potential overlapped batch
        if self.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(self.decode_producer_stream)
        self.wait_for_pending_backup()

        # Use kv_allocated_len (not seqlen): under speculative decoding the
        # allocator can over-allocate beyond the committed seqlen, and those
        # extra slots may carry stale mapping entries pointing at buffer slots
        # we just freed via free_hisparse_indices(all_hi). If left set, the
        # subsequent release_kv_cache -> allocator.free -> free_hisparse path
        # re-frees them (double-free into the page allocator's free list).
        allocated_len = req.kv_allocated_len
        compressed_len = allocated_len // self.compress_ratio

        # release memory -- only free actually-allocated buffer indices
        current_cap = int(self.req_device_buffer_size[req.req_pool_idx])
        if current_cap > 0:
            side_buf_hi = self.req_to_device_buffer[req.req_pool_idx, :current_cap]
            all_hi = torch.unique(side_buf_hi[side_buf_hi > 0])
            if all_hi.numel() > 0:
                self.token_to_kv_pool_allocator.free_hisparse_indices(all_hi)

        allocated_locs = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :allocated_len
        ]
        # Only clear the mapping when alloc_device_buffer was actually called
        # (current_cap > 0).  When current_cap == 0 the mapping still holds valid
        # hisparse indices that will be freed by the subsequent release_kv_cache →
        # cache_finished_req → free() → free_hisparse() path.
        if current_cap > 0:
            mapping_locs = (
                self.mem_pool_device.translate_loc_from_full_to_compressed(
                    allocated_locs
                )
                if self.is_dsv4_hisparse
                else allocated_locs
            )
            self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping[
                mapping_locs
            ] = 0

        host_indices = self.req_to_host_pool[req.req_pool_idx, :compressed_len]
        host_indices = host_indices[host_indices >= 0]
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)

        # clear req info
        self.req_device_buffer_tokens[:, req.req_pool_idx, :] = -1
        self.req_device_buffer_token_locs[:, req.req_pool_idx, :] = -1
        self.req_to_device_buffer[req.req_pool_idx, :] = 0
        self.req_device_buffer_size[req.req_pool_idx] = 0
        self.req_device_buffer_regular_size[req.req_pool_idx] = 0
        self.req_device_buffer_regular_size_gpu[req.req_pool_idx] = 0
        self.req_to_host_pool[req.req_pool_idx, :] = -1
        self.lru_slots[:, req.req_pool_idx, :].copy_(self._lru_init)
        self._skip_first_backup[req.req_pool_idx] = False
        self._exact_host_slots[req.req_pool_idx] = False

    def swap_in_selected_pages(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
    ) -> SwapResult:
        """Classify hit/miss, update LRU, assign slots. H2D copy is NOT done here.

        Returns:
            (top_k_device_locs, hit_device_locs, miss_device_locs,
             hit_count, miss_src_locs, miss_dst_locs)
        """
        # Ensure any pending D2H backup from finalize_accepted_tokens has completed
        # before reading host cache data.
        # Skip during CUDA graph capture mode since stream synchronization is incompatible
        if not get_is_capture_mode():
            device_module.current_stream().wait_event(self.d2h_finish_event)

        if req_pool_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"req_pool_indices dtype {req_pool_indices.dtype} is not int32 or int64 as expected"
            )
        if compressed_seq_lens.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"compressed_seq_lens dtype {compressed_seq_lens.dtype} is not int32 or int64 as expected"
            )
        if top_k_result.dtype != torch.int32:
            raise ValueError(
                f"top_k_result dtype {top_k_result.dtype} is not int32 as expected"
            )

        nvtx.range_push(f"hisparse::prepare_swap L{layer_id}")
        num_reqs = req_pool_indices.size(0)

        top_k_indices = self.top_k_device_locs_buffer[:num_reqs]
        top_k_indices.fill_(-1)
        hit_locs = self.hit_device_locs_buffer[:num_reqs]
        hit_locs.fill_(-1)
        miss_locs = self.miss_device_locs_buffer[:num_reqs]
        miss_locs.fill_(-1)
        hit_count = self.hit_count_buffer[:num_reqs]
        miss_src = self.miss_src_locs_buffer[:num_reqs]
        miss_dst = self.miss_dst_locs_buffer[:num_reqs]

        block_size = 1024
        prepare_swap_mla(
            top_k_tokens=top_k_result,
            device_buffer_tokens=self.req_device_buffer_tokens[layer_id],
            host_cache_locs=self.req_to_host_pool,
            device_buffer_locs=self.req_device_buffer_token_locs[layer_id],
            top_k_device_locs=top_k_indices,
            hit_device_locs=hit_locs,
            miss_device_locs=miss_locs,
            hit_count=hit_count,
            miss_src_locs=miss_src,
            miss_dst_locs=miss_dst,
            req_pool_indices=req_pool_indices,
            seq_lens=compressed_seq_lens,
            lru_slots=self.lru_slots[layer_id],
            item_size_bytes=self.item_size_bytes,
            num_top_k=self.top_k,
            hot_buffer_size=self.device_buffer_size,
            page_size=1,
            block_size=block_size,
            num_real_reqs=self.num_real_reqs,
            is_dsv4_layout=self.is_dsv4_hisparse,
        )
        nvtx.range_pop()
        return SwapResult(
            top_k_indices, hit_locs, miss_locs, hit_count, miss_src, miss_dst
        )

    def execute_h2d_async(
        self,
        miss_src_locs: torch.Tensor,
        miss_dst_locs: torch.Tensor,
        hit_count: torch.Tensor,
        layer_id: int,
    ) -> None:
        """Launch H2D copy on the transfer stream. Non-blocking."""
        # Skip during CUDA graph capture mode - H2D operations can't be captured
        if get_is_capture_mode():
            return
        nvtx.range_push(f"hisparse::h2d_async L{layer_id}")
        if logger.isEnabledFor(logging.DEBUG):
            hit = int(hit_count.sum().item())
            logger.debug(
                "hisparse H2D async: layer=%d hit=%d miss=%d total=%d",
                layer_id,
                hit,
                miss_src_locs.numel() - hit,
                miss_src_locs.numel(),
            )
        self.h2d_start_event.record()
        self.transfer_stream.wait_event(self.h2d_start_event)

        with device_module.stream(self.transfer_stream):
            execute_h2d_copy_mla(
                miss_src_locs=miss_src_locs,
                miss_dst_locs=miss_dst_locs,
                hit_count=hit_count,
                host_cache=self.mem_pool_host.kv_buffer[layer_id],
                device_buffer=self.mem_pool_device.kv_buffer[layer_id],
                item_size_bytes=self.item_size_bytes,
                num_top_k=self.top_k,
                hot_buffer_size=self.device_buffer_size,
                block_size=1024,
                num_real_reqs=self.num_real_reqs,
                is_dsv4_layout=self.is_dsv4_hisparse,
            )
        self.h2d_finish_event.record(self.transfer_stream)
        nvtx.range_pop()

    def execute_h2d_sync(
        self,
        miss_src_locs: torch.Tensor,
        miss_dst_locs: torch.Tensor,
        hit_count: torch.Tensor,
        layer_id: int,
    ) -> None:
        """Synchronous H2D copy on the current stream. Used by non-dual-attention backends."""
        nvtx.range_push(f"hisparse::h2d_sync L{layer_id}")
        execute_h2d_copy_mla(
            miss_src_locs=miss_src_locs,
            miss_dst_locs=miss_dst_locs,
            hit_count=hit_count,
            host_cache=self.mem_pool_host.kv_buffer[layer_id],
            device_buffer=self.mem_pool_device.kv_buffer[layer_id],
            item_size_bytes=self.item_size_bytes,
            num_top_k=self.top_k,
            hot_buffer_size=self.device_buffer_size,
            block_size=1024,
            num_real_reqs=self.num_real_reqs,
            is_dsv4_layout=self.is_dsv4_hisparse,
        )
        nvtx.range_pop()

    def wait_h2d(self) -> None:
        """Block current stream until H2D transfer completes."""
        # Skip during CUDA graph capture mode
        if get_is_capture_mode():
            return
        nvtx.range_push("hisparse::wait_h2d")
        device_module.current_stream().wait_event(self.h2d_finish_event)
        nvtx.range_pop()

    # --- Speculative decoding integration ---

    def get_draft_device_slots(
        self,
        req_pool_indices: torch.Tensor,
        num_tokens_per_req: int,
    ) -> torch.Tensor:
        """Return device buffer physical KV locations from the extra page.

        The extra page occupies buffer positions [device_buffer_size .. padded_buffer_size-1].
        Position device_buffer_size is the newest-decode slot; positions
        device_buffer_size+1 .. padded_buffer_size-1 are available for draft tokens.

        Returns:
            (bs * num_tokens_per_req,) int64 tensor of physical KV buffer row indices.
        """
        req_indices_cpu = req_pool_indices.cpu()
        regular_sizes = self.req_device_buffer_regular_size[req_indices_cpu]
        alloc_sizes = self.req_device_buffer_size[req_indices_cpu]
        if torch.any(regular_sizes + 1 + num_tokens_per_req > alloc_sizes):
            raise ValueError(
                f"Requested {num_tokens_per_req} draft slots but at least one "
                "request does not have enough speculative extra-page capacity"
            )
        starts = self.req_device_buffer_regular_size_gpu[req_pool_indices] + 1
        offsets = torch.arange(
            num_tokens_per_req, dtype=torch.int64, device=req_pool_indices.device
        )
        cols = starts.unsqueeze(1) + offsets.unsqueeze(0)
        return self.req_to_device_buffer[req_pool_indices.unsqueeze(1), cols].reshape(
            -1
        )

    def get_draft_device_slots_variable(
        self,
        req_pool_indices: torch.Tensor,
        tokens_per_req: torch.Tensor,
    ) -> torch.Tensor:
        """Like get_draft_device_slots, but each request can need a different count."""
        max_tokens = (
            int(tokens_per_req.max().item()) if tokens_per_req.numel() > 0 else 0
        )
        req_indices_cpu = req_pool_indices.cpu()
        regular_sizes = self.req_device_buffer_regular_size[req_indices_cpu]
        alloc_sizes = self.req_device_buffer_size[req_indices_cpu]
        if torch.any(regular_sizes + 1 + max_tokens > alloc_sizes):
            raise ValueError(
                f"Max per-request draft slots ({max_tokens}) exceeds at least "
                "one request's speculative extra-page capacity"
            )
        bs = req_pool_indices.shape[0]
        if bs == 0:
            return torch.empty(0, dtype=torch.int64, device=req_pool_indices.device)

        counts = tokens_per_req.to(torch.int32)
        starts = self.req_device_buffer_regular_size_gpu[req_pool_indices] + 1
        col_offsets = torch.arange(max_tokens, device=req_pool_indices.device)
        cols = starts.unsqueeze(1) + col_offsets.unsqueeze(0)
        gathered = self.req_to_device_buffer[req_pool_indices.unsqueeze(1), cols]
        mask = col_offsets.unsqueeze(0) < counts.unsqueeze(1)
        return gathered[mask]

    def finalize_accepted_tokens(
        self,
        req_pool_indices: torch.Tensor,
        accepted_cache_locs: torch.Tensor,
        accept_length: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None:
        """Transition accepted draft tokens from extra page into the coordinator lifecycle.

        After speculative verify, accepted tokens' KV lives in the extra page.
        This method:
        1. Backs up all accepted tokens' KV from device to host (batched).
        2. Moves the last accepted token per request to the newest slot.
        3. Clears device mapping for host-only tokens.
        4. Sets _skip_first_backup for each request.
        """
        # Skip during CUDA graph capture mode - host operations can't be captured
        if get_is_capture_mode():
            return
        if accepted_cache_locs.numel() == 0:
            return

        bs = len(req_pool_indices)
        total_accepted = accepted_cache_locs.numel()
        mapping = self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping

        # --- Step 1: Batched D2H backup for ALL accepted tokens ---
        all_device_locs = self.mem_pool_device._translate_loc_to_hisparse_device(
            accepted_cache_locs
        )
        n_per_req = accept_length + 1  # (bs,) actual counts
        n_per_req_cpu = n_per_req.cpu().tolist()
        seq_lens_cpu = seq_lens.cpu().tolist()
        req_pool_indices_cpu = req_pool_indices.cpu()

        host_locs_list = []
        offset = 0
        for i in range(bs):
            n = n_per_req_cpu[i]
            req_idx = int(req_pool_indices_cpu[i])
            accepted_locs = accepted_cache_locs[offset : offset + n]
            if self._exact_host_slots[req_idx]:
                post_seq_len = seq_lens_cpu[i]
                token_positions = torch.arange(
                    post_seq_len - n,
                    post_seq_len,
                    dtype=torch.int64,
                    device=self.device,
                )
                host_locs_list.append(
                    self._alloc_exact_host_or_existing(
                        req_idx,
                        token_positions,
                        accepted_locs,
                        "accepted draft backup",
                    )
                )
            else:
                allocated = self.mem_pool_host.alloc(n)
                if allocated is None:
                    logger.error(
                        "HiSparse: host alloc failed for %d accepted draft tokens",
                        n,
                    )
                    raise RuntimeError(
                        "HiSparse host alloc failed for accepted draft tokens "
                        f"(req_idx={req_idx}, count={n}); "
                        f"{self.host_pool_debug_info()}"
                    )
                host_locs_list.append(allocated)
            offset += n
        host_locs = torch.cat(host_locs_list)
        if host_locs.numel() != total_accepted:
            raise RuntimeError(
                "HiSparse accepted draft backup produced mismatched host locations: "
                f"host={host_locs.numel()}, accepted={total_accepted}"
            )
        host_locs = host_locs.to(device=self.device)

        self.transfer_stream.wait_event(device_module.current_stream().record_event())
        with device_module.stream(self.transfer_stream):
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_locs,
                all_device_locs.contiguous(),
                io_backend="kernel",
            )
        self.d2h_finish_event.record(self.transfer_stream)
        # --- Step 2: Per-request bookkeeping via CUDA kernel ---
        # (host pool scatter, device mapping update, needs_move computation)
        cumsum = self.cumsum_buffer[: bs + 1]
        cumsum[0] = 0
        cumsum[1:] = torch.cumsum(n_per_req, dim=0)

        newest_buf_pos = self.req_device_buffer_regular_size_gpu[req_pool_indices]
        all_newest_phys = self.req_to_device_buffer[req_pool_indices, newest_buf_pos]

        needs_move = self.needs_move_buffer[:bs]
        last_accepted_device = self.last_accepted_device_buffer[:bs]
        newest_slot_device = self.newest_slot_device_buffer[:bs]

        finalize_accepted(
            self.req_to_host_pool,
            mapping,
            req_pool_indices,
            seq_lens.to(torch.int32),
            host_locs,
            accepted_cache_locs,
            all_device_locs,
            all_newest_phys,
            cumsum,
            needs_move,
            last_accepted_device,
            newest_slot_device,
        )

        # For short sequences the regular hot buffer contains the full context.
        # EAGLE target-verify uses the extend attention path and translates the
        # logical page table directly, so accepted tokens must keep a visible
        # logical->device mapping instead of becoming host-only.
        short_req_rows = []
        remap_locs = []
        remap_device_locs = []
        move_src_locs = []
        move_dst_locs = []
        offset = 0
        for i in range(bs):
            n = n_per_req_cpu[i]
            post_seq_len = seq_lens_cpu[i]
            req_idx = int(req_pool_indices_cpu[i])
            regular_size = int(self.req_device_buffer_regular_size[req_idx])
            if post_seq_len <= regular_size:
                token_positions = torch.arange(
                    post_seq_len - n,
                    post_seq_len,
                    dtype=torch.int64,
                    device=self.device,
                )
                dst_locs = self.req_to_device_buffer[req_idx, token_positions]
                accepted_locs = accepted_cache_locs[offset : offset + n]
                src_locs = all_device_locs[offset : offset + n]

                short_req_rows.append(i)
                remap_locs.append(accepted_locs)
                remap_device_locs.append(dst_locs)
                move_src_locs.append(src_locs)
                move_dst_locs.append(dst_locs)
            offset += n

        if short_req_rows:
            short_req_rows = torch.tensor(
                short_req_rows, dtype=torch.int64, device=self.device
            )
            needs_move[short_req_rows] = 0

            remap_locs = torch.cat(remap_locs)
            remap_device_locs = torch.cat(remap_device_locs)
            mapping[remap_locs] = remap_device_locs

            move_src_locs = torch.cat(move_src_locs)
            move_dst_locs = torch.cat(move_dst_locs)
            move_mask = move_src_locs != move_dst_locs
            if move_mask.any():
                self.mem_pool_device.transfer_values_on_device(
                    dst_indices=move_dst_locs[move_mask],
                    src_indices=move_src_locs[move_mask],
                )

        # _skip_first_backup: CPU list, must iterate (cheap, bs is small)
        for i in range(bs):
            self._skip_first_backup[int(req_pool_indices_cpu[i])] = True

        # --- Step 3: Batched KV move for last-accepted → newest slot ---
        move_mask = needs_move.bool()
        if move_mask.any():
            idx = torch.where(move_mask)[0].to(torch.int64)
            self.mem_pool_device.transfer_values_on_device(
                dst_indices=newest_slot_device[idx],
                src_indices=last_accepted_device[idx],
            )

        # The next speculative verify reuses the same extra-page slots. Ensure
        # the async D2H backup has finished before those slots can be overwritten.
        device_module.current_stream().wait_event(self.d2h_finish_event)
