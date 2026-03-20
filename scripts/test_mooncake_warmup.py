#!/usr/bin/env python3
"""
Standalone script to test MooncakeDistributedStore put/get.
Mirrors the warmup() logic in mooncake_store.py and prints detailed diagnostics.

Usage (run on the server with mooncake installed):
    python test_mooncake_warmup.py
"""

import os
import uuid
import sys
import socket

# ── hardcoded config (from k8s yaml) ────────────────────────────────────────
os.environ.setdefault("MOONCAKE_MASTER",               "mk-mooncake-master:50051")
os.environ.setdefault("MOONCAKE_TE_META_DATA_SERVER",  "etcd://26.5.65.103:2379;26.5.66.145:2379;26.5.67.232:2379")
os.environ.setdefault("MOONCAKE_GLOBAL_SEGMENT_SIZE",  "0")
os.environ.setdefault("MOONCAKE_PROTOCOL",             "rdma")
os.environ.setdefault("MOONCAKE_DEVICE",               "mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3")

# ── read env (env vars override hardcoded values above) ─────────────────────
master            = os.environ["MOONCAKE_MASTER"]
protocol          = os.environ["MOONCAKE_PROTOCOL"]
device_name       = os.environ["MOONCAKE_DEVICE"]
metadata_server   = os.environ["MOONCAKE_TE_META_DATA_SERVER"]
local_hostname    = (os.environ.get("MOONCAKE_LOCAL_HOSTNAME")
                     or os.environ.get("LOCAL_HOSTNAME", ""))
_global_seg_raw   = os.environ.get("MOONCAKE_GLOBAL_SEGMENT_SIZE", "0")
global_seg_size   = int(_global_seg_raw) if int(_global_seg_raw) > 0 else 2 * 1024 ** 3

print("=== Mooncake warmup test ===")
print(f"  MOONCAKE_MASTER:              {master!r}")
print(f"  MOONCAKE_PROTOCOL:            {protocol!r}")
print(f"  MOONCAKE_DEVICE:              {device_name!r}")
print(f"  MOONCAKE_TE_META_DATA_SERVER: {metadata_server!r}")
print(f"  LOCAL_HOSTNAME:               {local_hostname!r}")
print(f"  global_segment_size:          {global_seg_size}")
print()

if not master:
    print("ERROR: MOONCAKE_MASTER is not set.")
    sys.exit(1)

if not local_hostname:
    local_hostname = socket.gethostname()
    print(f"  (local_hostname fallback to socket.gethostname: {local_hostname!r})")

# ── import mooncake ──────────────────────────────────────────────────────────
try:
    from mooncake.store import MooncakeDistributedStore
    print("✅ mooncake imported OK")
except ImportError as e:
    print(f"❌ import mooncake failed: {e}")
    sys.exit(1)

# ── setup ────────────────────────────────────────────────────────────────────
DEFAULT_LOCAL_BUFFER_SIZE = 16 * 1024 * 1024  # 16 MB

store = MooncakeDistributedStore()
print(f"\nstore type: {type(store)}")

ret_code = store.setup(
    local_hostname,        # client_hostname
    metadata_server,       # metadata_server
    global_seg_size,       # global_segment_size
    DEFAULT_LOCAL_BUFFER_SIZE,
    protocol,              # protocol
    device_name,           # device_name
    master,                # master_server_address
    None,                  # transfer_engine (None = create new)
)
print(f"store.setup() -> ret_code={ret_code}")
if ret_code:
    print("❌ setup FAILED")
    sys.exit(1)
print("✅ setup OK")

# ── warmup put ────────────────────────────────────────────────────────────────
warmup_key   = "sglang_mooncake_store_warmup_key" + uuid.uuid4().hex
warmup_value = bytes(4 * 1024)   # 4 KB of zeros

print(f"\nTrying put: key={warmup_key!r}, value_len={len(warmup_value)}")
ret = store.put(warmup_key, warmup_value)
print(f"store.put() -> ret={ret}  (expected 0)")
if ret != 0:
    print("❌ put FAILED")
    # Try with different value sizes to narrow down
    for sz in [1, 64, 512, 1024, 4096, 65536]:
        r2 = store.put(warmup_key + "_sz" + str(sz), bytes(sz))
        print(f"  put size={sz:6d} -> ret={r2}")
    sys.exit(1)

ret = store.is_exist(warmup_key)
print(f"store.is_exist() -> ret={ret}  (expected 1)")

got = store.get(warmup_key)
match = (got == warmup_value)
print(f"store.get() -> len={len(got) if got else None}, match={match}")

if match:
    print("\n✅ warmup PASSED")
else:
    print("\n❌ warmup FAILED: get value mismatch")
    sys.exit(1)
