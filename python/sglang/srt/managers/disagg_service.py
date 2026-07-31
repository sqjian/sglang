"""Start bootstrap/kv-store-related server"""

import os

from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    KVClassType,
    TransferBackend,
    get_kv_class,
)
from sglang.srt.server_args import ServerArgs


def start_disagg_service(
    server_args: ServerArgs,
):
    # Start kv bootstrap server on prefill
    disagg_mode = DisaggregationMode(server_args.disaggregation_mode)
    transfer_backend = TransferBackend(server_args.disaggregation_transfer_backend)

    if disagg_mode == DisaggregationMode.PREFILL:
        # only start bootstrap server on prefill tm
        kv_bootstrap_server_class = get_kv_class(
            transfer_backend, KVClassType.BOOTSTRAP_SERVER
        )
        bootstrap_server = kv_bootstrap_server_class(
            host=server_args.host,
            port=server_args.disaggregation_bootstrap_port,
        )
        _maybe_create_ascend_config_store(
            server_args=server_args,
            transfer_backend=transfer_backend,
        )

        return bootstrap_server


def start_rust_disagg_service(server_args: ServerArgs):
    """Start the native bootstrap registry for an embedded-Rust prefill node.

    The returned object owns the listener and must remain referenced. Decode
    and unified roles are no-ops.
    """
    disagg_mode = DisaggregationMode(server_args.disaggregation_mode)
    if disagg_mode != DisaggregationMode.PREFILL:
        return None

    # Lazy import: the compiled extension exists only in Rust-server builds.
    from sglang.srt.server._core import BootstrapServer

    bootstrap_server = BootstrapServer(
        host=server_args.host,
        port=server_args.disaggregation_bootstrap_port,
    )
    _maybe_create_ascend_config_store(
        server_args=server_args,
        transfer_backend=TransferBackend(server_args.disaggregation_transfer_backend),
    )
    return bootstrap_server


def _maybe_create_ascend_config_store(
    server_args: ServerArgs,
    transfer_backend: TransferBackend,
) -> None:
    if not (server_args.node_rank == 0 and transfer_backend == TransferBackend.ASCEND):
        return

    try:
        from memfabric_hybrid import create_config_store

        ascend_url = os.getenv("ASCEND_MF_STORE_URL")
        create_config_store(ascend_url)
    except Exception as error:
        raise RuntimeError(
            "Failed create mf store, invalid ascend_url. " f"With exception {error}"
        ) from error
