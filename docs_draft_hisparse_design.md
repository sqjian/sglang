# Draft-also-uses-HiSparse: design & implementation plan

Branch: `mtp_glm52_draft_hisparse`

## Goal

Make the EAGLE/MTP draft model attend through the **same HiSparse sparse
attention** as the target, instead of the current dense-over-HiSparse
workaround. The dense draft sees the full exact context while the HiSparse
target attends a sparse, host-offloaded subset; their next-token
distributions diverge as context grows, so acceptance fluctuates (0.24..0.74)
and stays below the dense baseline.

## Root cause (confirmed)

- The draft forward attaches the **target's** `HiSparseCoordinator`
  (`eagle_draft*_cuda_graph_runner._attach_hisparse_coordinator` pulls
  `target_model_runner.hisparse_coordinator`).
- That coordinator's `mem_pool_device` / `mem_pool_host` / per-layer device
  buffer state (`req_device_buffer_tokens`, `req_device_buffer_token_locs`,
  `lru_slots`, shape `[TARGET_layer_num, max_req, padded_buf]`) only cover the
  **target** layers.
- The draft MTP layer has `layer_num = 1` and a `layer_id = num_hidden_layers`
  (out of the target range). `swap_in_selected_pages(layer_id)` loads
  host->device into `mem_pool_device.kv_buffer[layer_id]` (target). The draft
  pool's device buffer is therefore never populated -> draft reads stale/uninit
  KV -> acceptance collapse (the 0.13 case; the dense workaround masked it but
  introduced the dense/sparse mismatch).

## Chosen approach: dedicated draft HiSparse store driven by the same coordinator

Keep ONE coordinator but give it a registered **draft KV store** with its own
per-layer state, mirroring all data movement on the same logical slot
decisions. Specifically the coordinator gains optional, draft-side parallels:

| target field                          | draft parallel                         |
|---------------------------------------|----------------------------------------|
| `mem_pool_device` (HiSparse pool)     | `draft_mem_pool_device`                |
| `mem_pool_host`                       | `draft_mem_pool_host`                  |
| `req_device_buffer_tokens` [Lt,...]   | `draft_req_device_buffer_tokens` [Ld,] |
| `req_device_buffer_token_locs`        | `draft_..._token_locs`                 |
| `lru_slots`                           | `draft_lru_slots`                      |
| `req_to_host_pool` (shared slot ids)  | shared (same host slot ids, own buffer)|
| `full_to_hisparse_device_index_mapping` (shared) | shared                      |

Shared (one slot space, identical residency decisions):
`req_to_token_pool`, `token_to_kv_pool_allocator`,
`full_to_hisparse_device_index_mapping`, `req_to_device_buffer`,
`req_to_host_pool` (host slot numbers; the draft host pool is a separate
tensor indexed by the same numbers).

### Data-movement points to mirror (each must also touch the draft store)

1. **swap_in_selected_pages(layer_id, ...)** — when the *draft* layer forwards,
   it calls swap-in with the draft layer id; the coordinator routes to the
   draft per-layer state + draft host/device buffers. Add a `store="target"|
   "draft"` selector (or detect by layer_id range) so the kernel reads/writes
   the draft tensors.
2. **Prefill staging / admit** (`admit_request_into_staging`,
   `collect_ready_reqs`, `backup_from_device_all_layer`) — the draft prefill KV
   must be staged to the draft host pool too. The draft model runs its own
   prefill (draft extend), producing draft KV that must be backed up for the
   evicted region.
3. **Decode backup** (`finalize_accepted_tokens` / `_spec` path,
   `_backup_device_locs_to_host`) — accepted-token draft KV -> draft host.
4. **Newest-token slot remap / move_kv_cache** — apply to draft pool too.
5. **Host alloc** (`alloc_paged_token_slots`) — the draft host pool shares the
   same `req_to_host_pool` slot numbers; either share allocation with the
   target (single alloc, both buffers use the number) or allocate in lockstep.

### Per-layer state sizing

The draft store's per-layer arrays are sized to the **draft** `layer_num`
(== 1 for MTP), indexed by `draft_layer_id - draft_start_layer`. Do NOT index
the target arrays with the draft layer id.

## Step plan (incremental, each independently testable on GPU)

1. **[DONE]** Add env flag `SGLANG_HISPARSE_DRAFT_SPARSE` (default False keeps
   the current dense workaround so nothing regresses). In
   `model_runner_kv_cache_mixin`, gate `_draft_dense_over_hisparse` on
   `not SGLANG_HISPARSE_DRAFT_SPARSE`. In sparse mode the draft keeps its
   `HiSparseDSATokenToKVPool` sharing the mapping (the existing 871-877 path).
2. **[DONE]** Build the draft store inside the coordinator: `HiSparseDraftStore`
   (draft host pool `MLATokenToKVPoolHost` + draft per-layer buffers sized to
   the draft layer_num + independent `req_to_host_pool`). Provide
   `HiSparseCoordinator.register_draft_store(draft_pool)`, called from
   `EagleWorkerV2._maybe_register_hisparse_draft_store()` after draft backends
   init. DSV4-compressed HiSparse raises NotImplementedError for now.
3. **[DONE]** Route `swap_in_selected_pages` to the draft store for draft-layer
   forwards (add a `store` selector; the draft attention calls swap-in with its
   own layer id -> draft buffers/pools).
4. **[DONE]** Mirror prefill staging + decode backup + newest-slot remap to the
   draft store (draft prefill KV -> draft host; accepted draft KV -> draft host).
5. **[TODO]** Validate on GPU: with the flag on, PHYS-SLOT debug shows
   draft/verify agree, `full_to_hisparse_device_index_mapping[attended] != 0`
   for the draft, acceptance no longer length-dependent.

### Status

- Steps 1-4 implemented and compile-clean (no GPU validation yet).
  - Step 3: `swap_in_selected_pages` now takes `mem_pool_device=` and routes to
    `self._draft_store` (with a pool-local layer id) when the caller's pool is
    the draft pool. All `dsa_backend` call sites pass
    `mem_pool_device=self.token_to_kv_pool`.
  - Step 4: all coordinator data-movement points now mirror to `_draft_store`
    when it is registered (see "Design fork resolved" below).

### Design fork resolved: option B (shared slot bookkeeping)

We chose the **shared-mapping** variant. The draft pool shares the target's
`full_to_hisparse_device_index_mapping`, the physical device-slot allocation
(`req_to_device_buffer`, `req_device_buffer_size`), and the host-slot
allocation (`req_to_host_pool`). Only the KV **byte stores**
(`mem_pool_device` / `mem_pool_host`) and the per-(layer, req, slot) swap-in
**residency state** (`req_device_buffer_tokens`, `req_device_buffer_token_locs`,
`lru_slots`) are draft-specific. `register_draft_store` asserts the draft pool's
mapping IS the allocator's shared tensor, so `set_mla_kv_buffer(loc)` translates
draft writes to the SAME physical slot the target uses. Read and write therefore
land on the same slot; the residency state diverges only because each store
loads its own top-k into the hot buffer.

### Step 4 (DONE) — coordinator mirror points

All implemented in `hisparse_coordinator.py`, each gated on
`self._draft_store is not None`:

- **KV-byte movement** (same shared host/device slots): `admit_request_into_staging`
  (prefill stage), `_eager_backup_previous_token` + `_backup_device_locs_to_host`
  (decode backup), `_preload_to_device_buffer` (direct-admit preload),
  `finalize_accepted_tokens` (`transfer_values_on_device` newest-slot move).
- **Per-layer occupancy baseline**: `alloc_device_buffer`, `_grow_device_buffers`,
  `_ensure_padded_buffer`, `admit_request_direct` (empty-buffer reset),
  `map_last_loc_to_buffer` + `finalize_accepted_tokens` (newest-token slot),
  `get_draft_device_slots` / `get_draft_device_slots_variable` (draft-proposal
  extra-page residency).
- `prepare_verify_slots_spec_v2` is target-verify-only (the draft does not run
  during verify) and is intentionally NOT mirrored.

## Open questions

- Does the draft MTP layer run its own DSA indexer (own top_k) or reuse the
  target's selection? If own top_k, the draft is self-consistent (preferred).
- Memory budget: a draft host pool adds host memory ~ (draft_layer_num /
  target_layer_num) x target host pool. For 1 MTP layer this is small.
- CUDA-graph safety of the extra draft swap-in path (mirror the target's
  pre-allocated buffers + `num_real_reqs`).
