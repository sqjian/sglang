"""
HiCache Protocol Manager.

This module provides centralized management of HiCache protocol state,
including seq_id allocation, decision/outcome storage, alignment tracking,
and timeout handling.

Reference: docs/specs/op-cons/design.md
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple

from sglang.srt.mem_cache.hicache_protocol import (
    GlobalFailedNotice,
    HiCacheAction,
    HiCacheDecision,
    HiCacheFinalState,
    HiCacheFailureReason,
    HiCacheOutcome,
    HiCacheRequestState,
    ReleaseNotice,
)

logger = logging.getLogger(__name__)


@dataclass
class HiCacheProtocolConfig:
    """Configuration for HiCache protocol timeouts and behavior."""

    # Timeout for waiting for local prefetch to complete (seconds)
    prefetch_complete_timeout: float = 30.0

    # Timeout for waiting for previous rank's outcome (seconds)
    prev_rank_outcome_timeout: float = 10.0

    # Timeout for outcome alignment (seconds)
    alignment_timeout: float = 5.0

    # Whether to enable protocol (disabled when pp_size == 1)
    enabled: bool = True


class HiCacheProtocolManager:
    """
    Centralized manager for HiCache PP consistency protocol.

    This class handles:
    - seq_id allocation for decisions (monotonically increasing per request)
    - Decision and outcome storage and retrieval
    - Cross-rank alignment state tracking
    - Timeout and failure path handling
    - Deduplication of decisions and outcomes

    Note: This manager does not directly execute IO or tree operations.
    It only manages protocol state and consistency checks.
    """

    def __init__(
        self,
        pp_rank: int,
        pp_size: int,
        config: Optional[HiCacheProtocolConfig] = None,
    ):
        """
        Initialize the protocol manager.

        Args:
            pp_rank: This rank's position in the PP pipeline.
            pp_size: Total number of PP ranks.
            config: Protocol configuration (timeouts, etc.).
        """
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.config = config or HiCacheProtocolConfig()

        # Disable protocol for single-rank case
        if pp_size == 1:
            self.config.enabled = False

        self._lock = threading.Lock()

        # Per-request state tracking
        # Key: request_id
        self._request_states: Dict[str, HiCacheRequestState] = {}

        # seq_id counters per request (only used by rank 0)
        # Key: request_id, Value: next seq_id to allocate
        self._seq_id_counters: Dict[str, int] = {}

        # Decision storage
        # Key: (request_id, seq_id)
        self._decisions: Dict[Tuple[str, int], HiCacheDecision] = {}

        # Outcome storage
        # Key: (request_id, seq_id, source_pp_rank)
        self._outcomes: Dict[Tuple[str, int, int], HiCacheOutcome] = {}

        # Processed decision/outcome tracking for deduplication
        # Key: (request_id, seq_id)
        self._processed_decisions: Set[Tuple[str, int]] = set()
        self._processed_outcomes: Set[Tuple[str, int, int]] = set()

        # Pending alignments (requests waiting for outcome alignment)
        # Key: (request_id, seq_id), Value: timestamp when alignment started
        self._pending_alignments: Dict[Tuple[str, int], float] = {}

    @property
    def is_rank0(self) -> bool:
        """Check if this is rank 0 (the control plane leader)."""
        return self.pp_rank == 0

    @property
    def is_enabled(self) -> bool:
        """Check if protocol is enabled."""
        return self.config.enabled

    # =========================================================================
    # seq_id Management (rank 0 only)
    # =========================================================================

    def allocate_seq_id(self, request_id: str) -> int:
        """
        Allocate a new seq_id for a request (rank 0 only).

        Each HiCache control action (including retries) gets a new seq_id.
        seq_ids are monotonically increasing per request.

        Args:
            request_id: The request identifier.

        Returns:
            The allocated seq_id.

        Raises:
            RuntimeError: If called on a non-rank-0 node.
        """
        if not self.is_rank0:
            raise RuntimeError("Only rank 0 can allocate seq_ids")

        with self._lock:
            if request_id not in self._seq_id_counters:
                self._seq_id_counters[request_id] = 0

            seq_id = self._seq_id_counters[request_id]
            self._seq_id_counters[request_id] += 1

            # Update request state
            state = self._get_or_create_request_state(request_id)
            state.latest_decision_seq_id = seq_id

            return seq_id

    # =========================================================================
    # Decision Management
    # =========================================================================

    def store_decision(self, decision: HiCacheDecision) -> bool:
        """
        Store a decision.

        Args:
            decision: The decision to store.

        Returns:
            True if stored (new), False if duplicate.
        """
        key = (decision.request_id, decision.seq_id)

        with self._lock:
            if key in self._processed_decisions:
                logger.debug(
                    f"Duplicate decision ignored: {decision.request_id}:{decision.seq_id}"
                )
                return False

            self._decisions[key] = decision
            self._processed_decisions.add(key)

            # Update request state
            state = self._get_or_create_request_state(decision.request_id)
            if decision.seq_id > state.latest_decision_seq_id:
                state.latest_decision_seq_id = decision.seq_id
            state.pending_decision = decision

            return True

    def get_decision(
        self, request_id: str, seq_id: int
    ) -> Optional[HiCacheDecision]:
        """
        Get a stored decision.

        Args:
            request_id: The request identifier.
            seq_id: The sequence identifier.

        Returns:
            The decision if found, None otherwise.
        """
        with self._lock:
            return self._decisions.get((request_id, seq_id))

    def is_decision_newer(self, request_id: str, seq_id: int) -> bool:
        """
        Check if a decision's seq_id is newer than the latest for this request.

        Args:
            request_id: The request identifier.
            seq_id: The sequence identifier to check.

        Returns:
            True if seq_id is newer than the current latest.
        """
        with self._lock:
            state = self._request_states.get(request_id)
            if state is None:
                return True
            return seq_id > state.latest_decision_seq_id

    # =========================================================================
    # Outcome Management
    # =========================================================================

    def store_outcome(self, outcome: HiCacheOutcome) -> bool:
        """
        Store an outcome.

        Args:
            outcome: The outcome to store.

        Returns:
            True if stored (new), False if duplicate.
        """
        key = (outcome.request_id, outcome.seq_id, outcome.source_pp_rank)

        with self._lock:
            if key in self._processed_outcomes:
                logger.debug(
                    f"Duplicate outcome ignored: {outcome.request_id}:{outcome.seq_id} "
                    f"from rank {outcome.source_pp_rank}"
                )
                return False

            self._outcomes[key] = outcome
            self._processed_outcomes.add(key)

            # Update request state
            state = self._get_or_create_request_state(outcome.request_id)
            if outcome.source_pp_rank == self.pp_rank:
                state.local_outcome = outcome
                if outcome.seq_id > state.latest_outcome_seq_id:
                    state.latest_outcome_seq_id = outcome.seq_id
            elif outcome.source_pp_rank == self.pp_rank - 1:
                state.prev_rank_outcome = outcome

            return True

    def get_outcome(
        self, request_id: str, seq_id: int, source_pp_rank: int
    ) -> Optional[HiCacheOutcome]:
        """
        Get a stored outcome.

        Args:
            request_id: The request identifier.
            seq_id: The sequence identifier.
            source_pp_rank: The rank that produced the outcome.

        Returns:
            The outcome if found, None otherwise.
        """
        with self._lock:
            return self._outcomes.get((request_id, seq_id, source_pp_rank))

    def get_local_outcome(
        self, request_id: str, seq_id: int
    ) -> Optional[HiCacheOutcome]:
        """Get the local outcome for a request/seq_id."""
        return self.get_outcome(request_id, seq_id, self.pp_rank)

    def get_prev_rank_outcome(
        self, request_id: str, seq_id: int
    ) -> Optional[HiCacheOutcome]:
        """Get the previous rank's outcome for a request/seq_id."""
        if self.is_rank0:
            return None
        return self.get_outcome(request_id, seq_id, self.pp_rank - 1)

    # =========================================================================
    # Alignment Checking
    # =========================================================================

    def check_outcome_alignment(
        self, request_id: str, seq_id: int
    ) -> Tuple[bool, Optional[HiCacheFailureReason]]:
        """
        Check if outcomes are aligned between this rank and the previous rank.

        For rank 0, this always returns True (no alignment needed).
        For other ranks, this compares local outcome with prev_rank_outcome.

        Args:
            request_id: The request identifier.
            seq_id: The sequence identifier.

        Returns:
            Tuple of (is_aligned, failure_reason).
            If aligned, failure_reason is None.
            If not aligned, failure_reason indicates the mismatch type.
        """
        if self.is_rank0:
            return True, None

        with self._lock:
            local = self.get_local_outcome(request_id, seq_id)
            prev = self.get_prev_rank_outcome(request_id, seq_id)

            if local is None or prev is None:
                # Not yet ready for alignment
                return False, None

            # Check alignment criteria
            if local.final_state != prev.final_state:
                logger.warning(
                    f"Alignment mismatch: {request_id}:{seq_id} "
                    f"final_state {local.final_state} vs {prev.final_state}"
                )
                return False, HiCacheFailureReason.ALIGNMENT_MISMATCH

            if local.completed_tokens != prev.completed_tokens:
                logger.warning(
                    f"Alignment mismatch: {request_id}:{seq_id} "
                    f"completed_tokens {local.completed_tokens} vs {prev.completed_tokens}"
                )
                return False, HiCacheFailureReason.ALIGNMENT_MISMATCH

            if local.applied_tokens != prev.applied_tokens:
                logger.warning(
                    f"Alignment mismatch: {request_id}:{seq_id} "
                    f"applied_tokens {local.applied_tokens} vs {prev.applied_tokens}"
                )
                return False, HiCacheFailureReason.ALIGNMENT_MISMATCH

            logger.debug(
                f"Outcome aligned for {request_id}:{seq_id} "
                f"state={local.final_state.value}, completed={local.completed_tokens}"
            )
            return True, None

    def is_ready_for_forward(self, request_id: str, seq_id: int) -> bool:
        """
        Check if a request is ready to enter forward.

        Requirements:
        - Local outcome must be completed
        - For rank > 0: outcome must be aligned with previous rank

        Args:
            request_id: The request identifier.
            seq_id: The sequence identifier.

        Returns:
            True if ready for forward.
        """
        with self._lock:
            state = self._request_states.get(request_id)
            if state is None:
                return False

            # Check terminal state
            if state.terminal_state != "none":
                return False

            # Check outcome is complete
            if state.latest_outcome_seq_id < seq_id:
                return False

            # For rank > 0, check alignment
            if not self.is_rank0:
                aligned, _ = self.check_outcome_alignment(request_id, seq_id)
                return aligned

            return True

    # =========================================================================
    # Timeout Management
    # =========================================================================

    def start_alignment_wait(self, request_id: str, seq_id: int):
        """Start tracking alignment wait time."""
        with self._lock:
            self._pending_alignments[(request_id, seq_id)] = time.monotonic()

    def check_alignment_timeout(self, request_id: str, seq_id: int) -> bool:
        """
        Check if alignment wait has timed out.

        Returns:
            True if timed out.
        """
        with self._lock:
            start_time = self._pending_alignments.get((request_id, seq_id))
            if start_time is None:
                return False

            elapsed = time.monotonic() - start_time
            return elapsed > self.config.alignment_timeout

    def end_alignment_wait(self, request_id: str, seq_id: int):
        """Stop tracking alignment wait time."""
        with self._lock:
            self._pending_alignments.pop((request_id, seq_id), None)

    # =========================================================================
    # Request State Management
    # =========================================================================

    def _get_or_create_request_state(self, request_id: str) -> HiCacheRequestState:
        """Get or create request state (must hold lock)."""
        if request_id not in self._request_states:
            self._request_states[request_id] = HiCacheRequestState(
                request_id=request_id
            )
        return self._request_states[request_id]

    def get_request_state(self, request_id: str) -> Optional[HiCacheRequestState]:
        """Get request state."""
        with self._lock:
            return self._request_states.get(request_id)

    def mark_request_failed(self, request_id: str):
        """Mark a request as failed (terminal state)."""
        with self._lock:
            state = self._get_or_create_request_state(request_id)
            state.terminal_state = "failed"

    def mark_request_released(self, request_id: str):
        """Mark a request as released (terminal state)."""
        with self._lock:
            state = self._get_or_create_request_state(request_id)
            state.terminal_state = "released"

    def is_request_terminal(self, request_id: str) -> bool:
        """Check if request is in terminal state."""
        with self._lock:
            state = self._request_states.get(request_id)
            if state is None:
                return False
            return state.terminal_state != "none"

    # =========================================================================
    # Cleanup
    # =========================================================================

    def cleanup_request(self, request_id: str):
        """
        Clean up all state for a request.

        Called when a request completes, fails, or is aborted.
        """
        with self._lock:
            # Remove request state
            self._request_states.pop(request_id, None)
            self._seq_id_counters.pop(request_id, None)

            # Remove decisions and outcomes for this request
            decision_keys = [k for k in self._decisions if k[0] == request_id]
            for key in decision_keys:
                self._decisions.pop(key, None)

            outcome_keys = [k for k in self._outcomes if k[0] == request_id]
            for key in outcome_keys:
                self._outcomes.pop(key, None)

            # Remove from processed sets
            processed_decision_keys = [
                k for k in self._processed_decisions if k[0] == request_id
            ]
            for key in processed_decision_keys:
                self._processed_decisions.discard(key)

            processed_outcome_keys = [
                k for k in self._processed_outcomes if k[0] == request_id
            ]
            for key in processed_outcome_keys:
                self._processed_outcomes.discard(key)

            # Remove pending alignments
            alignment_keys = [
                k for k in self._pending_alignments if k[0] == request_id
            ]
            for key in alignment_keys:
                self._pending_alignments.pop(key, None)

    def reset(self):
        """Reset all protocol state."""
        with self._lock:
            self._request_states.clear()
            self._seq_id_counters.clear()
            self._decisions.clear()
            self._outcomes.clear()
            self._processed_decisions.clear()
            self._processed_outcomes.clear()
            self._pending_alignments.clear()

    # =========================================================================
    # Helper Methods for Creating Protocol Messages
    # =========================================================================

    def create_decision(
        self,
        request_id: str,
        action: HiCacheAction,
        planned_tokens: int,
        token_ids: list,
        prefix_length: int,
        last_hash: Optional[str] = None,
    ) -> HiCacheDecision:
        """
        Create a new HiCacheDecision (rank 0 only).

        This allocates a seq_id and creates the decision object.
        """
        seq_id = self.allocate_seq_id(request_id)
        decision = HiCacheDecision(
            request_id=request_id,
            seq_id=seq_id,
            action=action,
            planned_tokens=planned_tokens,
            token_ids=token_ids,
            prefix_length=prefix_length,
            last_hash=last_hash,
        )
        self.store_decision(decision)
        return decision

    def create_outcome(
        self,
        request_id: str,
        seq_id: int,
        final_state: HiCacheFinalState,
        completed_tokens: int,
        applied_tokens: int,
        local_prefix_length: int,
        failure_reason: Optional[HiCacheFailureReason] = None,
    ) -> HiCacheOutcome:
        """Create a new HiCacheOutcome for this rank."""
        outcome = HiCacheOutcome(
            request_id=request_id,
            seq_id=seq_id,
            source_pp_rank=self.pp_rank,
            final_state=final_state,
            completed_tokens=completed_tokens,
            applied_tokens=applied_tokens,
            local_prefix_length=local_prefix_length,
            failure_reason=failure_reason,
        )
        self.store_outcome(outcome)
        return outcome

    def create_global_failed_notice(
        self,
        request_id: str,
        seq_id: int,
        reason: str,
        source_pp_rank: int,
    ) -> GlobalFailedNotice:
        """Create a GlobalFailedNotice."""
        from sglang.srt.mem_cache.hicache_protocol import GlobalFailedReason

        return GlobalFailedNotice(
            request_id=request_id,
            seq_id=seq_id,
            reason=GlobalFailedReason(reason),
            source_pp_rank=source_pp_rank,
        )

    def create_release_notice(
        self, request_id: str, seq_id: int
    ) -> ReleaseNotice:
        """Create a ReleaseNotice."""
        return ReleaseNotice(request_id=request_id, seq_id=seq_id)

    # =========================================================================
    # Superseded/Lazy Cancel Support
    # =========================================================================

    def mark_old_prefetch_superseded(self, request_id: str, new_seq_id: int):
        """
        Mark old prefetch operations as superseded.

        When a new decision arrives, old prefetches should not be used
        for tree updates or alignment.

        Args:
            request_id: The request identifier.
            new_seq_id: The new seq_id that supersedes older ones.
        """
        with self._lock:
            # Find and mark older outcomes/decisions as superseded
            # The actual prefetch cancellation is lazy (done on IO completion)
            state = self._request_states.get(request_id)
            if state is not None and state.pending_decision is not None:
                old_seq_id = state.pending_decision.seq_id
                if old_seq_id < new_seq_id:
                    logger.debug(
                        f"Superseding old decision {request_id}:{old_seq_id} "
                        f"with new decision {request_id}:{new_seq_id}"
                    )

    # =========================================================================
    # Extended Timeout Management
    # =========================================================================

    def start_prefetch_wait(self, request_id: str, seq_id: int):
        """
        Start tracking prefetch wait time for a request.

        Called when a prefetch operation is initiated.
        """
        key = (request_id, seq_id, "prefetch")
        with self._lock:
            if not hasattr(self, "_prefetch_wait_times"):
                self._prefetch_wait_times: Dict[Tuple[str, int, str], float] = {}
            self._prefetch_wait_times[key] = time.monotonic()

    def check_prefetch_timeout(self, request_id: str, seq_id: int) -> bool:
        """
        Check if prefetch operation has timed out.

        Returns:
            True if prefetch has exceeded prefetch_complete_timeout.
        """
        key = (request_id, seq_id, "prefetch")
        with self._lock:
            if not hasattr(self, "_prefetch_wait_times"):
                return False
            start_time = self._prefetch_wait_times.get(key)
            if start_time is None:
                return False

            elapsed = time.monotonic() - start_time
            return elapsed > self.config.prefetch_complete_timeout

    def end_prefetch_wait(self, request_id: str, seq_id: int):
        """Stop tracking prefetch wait time."""
        key = (request_id, seq_id, "prefetch")
        with self._lock:
            if hasattr(self, "_prefetch_wait_times"):
                self._prefetch_wait_times.pop(key, None)

    def start_prev_rank_outcome_wait(self, request_id: str, seq_id: int):
        """
        Start tracking wait time for previous rank's outcome.

        Called when waiting for the previous rank to send its outcome.
        """
        key = (request_id, seq_id, "prev_outcome")
        with self._lock:
            if not hasattr(self, "_prev_outcome_wait_times"):
                self._prev_outcome_wait_times: Dict[Tuple[str, int, str], float] = {}
            self._prev_outcome_wait_times[key] = time.monotonic()

    def check_prev_rank_outcome_timeout(self, request_id: str, seq_id: int) -> bool:
        """
        Check if waiting for previous rank's outcome has timed out.

        Returns:
            True if wait has exceeded prev_rank_outcome_timeout.
        """
        key = (request_id, seq_id, "prev_outcome")
        with self._lock:
            if not hasattr(self, "_prev_outcome_wait_times"):
                return False
            start_time = self._prev_outcome_wait_times.get(key)
            if start_time is None:
                return False

            elapsed = time.monotonic() - start_time
            return elapsed > self.config.prev_rank_outcome_timeout

    def end_prev_rank_outcome_wait(self, request_id: str, seq_id: int):
        """Stop tracking wait time for previous rank's outcome."""
        key = (request_id, seq_id, "prev_outcome")
        with self._lock:
            if hasattr(self, "_prev_outcome_wait_times"):
                self._prev_outcome_wait_times.pop(key, None)

    def create_timeout_outcome(
        self,
        request_id: str,
        seq_id: int,
        timeout_reason: HiCacheFailureReason,
        local_prefix_length: int = 0,
    ) -> HiCacheOutcome:
        """
        Create a failed HiCacheOutcome due to timeout.

        Args:
            request_id: The request identifier.
            seq_id: The seq_id of the decision that timed out.
            timeout_reason: One of timeout, prev_rank_timeout, or alignment_timeout.
            local_prefix_length: The local prefix length at timeout.

        Returns:
            A failed HiCacheOutcome with the appropriate reason.
        """
        return self.create_outcome(
            request_id=request_id,
            seq_id=seq_id,
            final_state=HiCacheFinalState.failed,
            completed_tokens=0,
            applied_tokens=0,
            local_prefix_length=local_prefix_length,
            failure_reason=timeout_reason,
        )

    def check_all_timeouts(
        self, request_id: str, seq_id: int, local_prefix_length: int = 0
    ) -> Optional[HiCacheOutcome]:
        """
        Check all timeout conditions for a request.

        This is a convenience method that checks all timeout conditions
        and creates an appropriate failed outcome if any timeout occurs.

        Args:
            request_id: The request identifier.
            seq_id: The seq_id being checked.
            local_prefix_length: The local prefix length for the outcome.

        Returns:
            A failed HiCacheOutcome if any timeout occurred, None otherwise.
        """
        # Check prefetch timeout
        if self.check_prefetch_timeout(request_id, seq_id):
            logger.warning(
                f"Prefetch timeout for request {request_id}:{seq_id} "
                f"(>{self.config.prefetch_complete_timeout}s)"
            )
            self.end_prefetch_wait(request_id, seq_id)
            return self.create_timeout_outcome(
                request_id, seq_id, HiCacheFailureReason.timeout, local_prefix_length
            )

        # Check prev rank outcome timeout
        if self.check_prev_rank_outcome_timeout(request_id, seq_id):
            logger.warning(
                f"Prev rank outcome timeout for request {request_id}:{seq_id} "
                f"(>{self.config.prev_rank_outcome_timeout}s)"
            )
            self.end_prev_rank_outcome_wait(request_id, seq_id)
            return self.create_timeout_outcome(
                request_id,
                seq_id,
                HiCacheFailureReason.prev_rank_timeout,
                local_prefix_length,
            )

        # Check alignment timeout
        if self.check_alignment_timeout(request_id, seq_id):
            logger.warning(
                f"Alignment timeout for request {request_id}:{seq_id} "
                f"(>{self.config.alignment_timeout}s)"
            )
            self.end_alignment_wait(request_id, seq_id)
            return self.create_timeout_outcome(
                request_id,
                seq_id,
                HiCacheFailureReason.alignment_timeout,
                local_prefix_length,
            )

        return None

