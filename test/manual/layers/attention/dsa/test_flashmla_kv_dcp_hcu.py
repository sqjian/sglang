"""Multi-GPU FlashMLA-KV DCP correctness check for HCU BW1000.

Run on one BW1000 node with DCP size 2, 4, or 8, for example::

    torchrun --standalone --nproc-per-node 8 \
      test/manual/layers/attention/dsa/test_flashmla_kv_dcp_sm90.py

This test intentionally avoids model weights. It covers the production owner
mapping, FP8 KV layout, FlashMLA's sparse-index holes, natural-log to base-2
LSE normalization, the cross-rank online-softmax merge, and a case where all
but one DCP rank have zero local KV entries.
"""

import os

import torch
import torch.distributed as dist
from flash_mla import flash_mla_with_kvcache, get_mla_metadata

from sglang.kernels.ops.attention.dsa.quant_k_cache import quantize_k_cache
from sglang.kernels.ops.attention.fixup_zero_kv import fixup_zero_kv_rows
from sglang.srt.utils import is_hcu

LOG2_E = 1.4426950408889634
TOPK = 2048
NUM_HEADS = 64
HEAD_DIM = 576
VALUE_DIM = 512
PAGE_SIZE = 64


def _flashmla_sparse_decode(
    q: torch.Tensor,
    kv_bf16: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = kv_bf16.shape[0]
    padded_tokens = ((num_tokens + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE
    kv_padded = torch.zeros(
        padded_tokens, HEAD_DIM, dtype=torch.bfloat16, device=q.device
    )
    kv_padded[:num_tokens] = kv_bf16
    kv_fp8 = quantize_k_cache(kv_padded.view(-1, PAGE_SIZE, 1, HEAD_DIM))

    cache_seqlens = torch.tensor([TOPK], dtype=torch.int32, device=q.device)
    scheduler_metadata, num_splits = get_mla_metadata(
        cache_seqlens=cache_seqlens,
        num_q_tokens_per_head_k=NUM_HEADS,
        num_heads_k=1,
        num_heads_q=NUM_HEADS,
        is_fp8_kvcache=True,
        topk=TOPK,
    )
    out, natural_lse = flash_mla_with_kvcache(
        q=q.view(1, 1, NUM_HEADS, HEAD_DIM),
        k_cache=kv_fp8,
        block_table=torch.empty((1, 0), dtype=torch.int32, device=q.device),
        cache_seqlens=cache_seqlens,
        head_dim_v=VALUE_DIM,
        tile_scheduler_metadata=scheduler_metadata,
        num_splits=num_splits,
        softmax_scale=HEAD_DIM**-0.5,
        is_fp8_kvcache=True,
        indices=indices.view(1, 1, TOPK),
    )
    return out.view(1, NUM_HEADS, VALUE_DIM), natural_lse.view(1, NUM_HEADS)


def _merge_dcp_partials(
    local_out: torch.Tensor,
    local_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    world_size = dist.get_world_size()
    gathered_lse = [torch.empty_like(local_lse) for _ in range(world_size)]
    dist.all_gather(gathered_lse, local_lse)
    lse_stack = torch.stack(gathered_lse)
    ln_2 = torch.log(torch.tensor(2.0, device=local_lse.device))
    global_lse = torch.logsumexp(lse_stack * ln_2, dim=0) * LOG2_E
    local_weight = torch.exp2(local_lse - global_lse).unsqueeze(-1)
    merged_out = local_out.float() * local_weight
    dist.all_reduce(merged_out)
    return merged_out, global_lse


def _run_case(
    *,
    case_name: str,
    q: torch.Tensor,
    global_kv: torch.Tensor,
    topk_indices: torch.Tensor,
) -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Production owner rule: global slot g belongs to rank g % DCP, at local
    # row g // DCP. Unowned entries stay -1 holes for FlashMLA.
    local_kv = global_kv[rank::world_size].contiguous()
    owned = topk_indices.remainder(world_size) == rank
    local_indices = torch.where(
        owned, topk_indices // world_size, torch.full_like(topk_indices, -1)
    )
    local_out, local_natural_lse = _flashmla_sparse_decode(q, local_kv, local_indices)

    local_lse = local_natural_lse.float() * LOG2_E
    local_kv_count = owned.sum(dtype=torch.int32).view(1)
    fixup_zero_kv_rows(
        local_out,
        local_lse,
        local_kv_count,
        torch.tensor([0, 1], dtype=torch.int32, device=q.device),
        max_seq_len=1,
    )
    merged_out, global_lse = _merge_dcp_partials(local_out, local_lse)

    if rank == 0:
        reference_out, reference_natural_lse = _flashmla_sparse_decode(
            q, global_kv, topk_indices
        )
        reference_lse = reference_natural_lse.float() * LOG2_E
        torch.testing.assert_close(
            merged_out,
            reference_out.float(),
            atol=2e-2,
            rtol=2e-2,
        )
        torch.testing.assert_close(
            global_lse,
            reference_lse,
            atol=2e-3,
            rtol=2e-3,
        )
        max_out_error = (merged_out - reference_out.float()).abs().max().item()
        max_lse_error = (global_lse - reference_lse).abs().max().item()
        print(
            f"{case_name} passed: max_out_error={max_out_error:.6f}, "
            f"max_lse_error={max_lse_error:.6f}"
        )


@torch.inference_mode()
def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size not in (2, 4, 8):
        raise ValueError(f"This test requires 2, 4, or 8 ranks, got {world_size}.")

    device = torch.device("cuda", local_rank)
    capability = torch.cuda.get_device_capability(device)
    if not is_hcu() or capability != (9, 3):
        raise ValueError(
            "flashmla_kv DCP integration test requires HCU BW1000 "
            f"capability (9, 3); got is_hcu={is_hcu()} capability={capability}."
        )

    seq_len = TOPK * world_size
    if rank == 0:
        generator = torch.Generator(device=device).manual_seed(20260826)
        q = torch.randn(
            NUM_HEADS,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ).clamp_(-1, 1)
        global_kv = (
            torch.randn(
                seq_len,
                HEAD_DIM,
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            / 10
        ).clamp_(-1, 1)
        balanced_indices = torch.randperm(
            seq_len, dtype=torch.int32, device=device, generator=generator
        )[:TOPK]
        zero_local_indices = torch.arange(
            0,
            seq_len,
            world_size,
            dtype=torch.int32,
            device=device,
        )[:TOPK]
    else:
        q = torch.empty(NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
        global_kv = torch.empty(seq_len, HEAD_DIM, dtype=torch.bfloat16, device=device)
        balanced_indices = torch.empty(TOPK, dtype=torch.int32, device=device)
        zero_local_indices = torch.empty(TOPK, dtype=torch.int32, device=device)

    for tensor in (q, global_kv, balanced_indices, zero_local_indices):
        dist.broadcast(tensor, src=0)

    _run_case(
        case_name=f"FlashMLA KV DCP{world_size} balanced HCU BW1000",
        q=q,
        global_kv=global_kv,
        topk_indices=balanced_indices,
    )
    _run_case(
        case_name=f"FlashMLA KV DCP{world_size} zero-local-KV HCU BW1000",
        q=q,
        global_kv=global_kv,
        topk_indices=zero_local_indices,
    )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
