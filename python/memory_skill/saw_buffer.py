"""
SawRingBuffer — O(1) ring buffer for short-term screen observations (Saw).

Uses ``collections.deque(maxlen=capacity)`` for automatic eviction of the
oldest entries when the buffer overflows.  Provides:

  - O(1) ``put()``      — deque.append()
  - O(1) ``get_at_offset(heartbeat_index)`` — index arithmetic from oldest entry
  - O(n) ``get_all()``  — full snapshot

Design adapted from OpenHeart's original SawRingBuffer, but capacity is
measured in *entry count* (default 1000) rather than time window.  Time-based
windowing is achieved implicitly through the heartbeat_index frequency.
"""

from __future__ import annotations

from collections import deque

from memory_skill.contracts import MemorySkillConfig, SawEntry


class SawRingBuffer:
    """A fixed-capacity ring buffer for ``SawEntry`` entries.

    Entries are stored in insertion order.  When the buffer reaches capacity,
    the oldest entry is **silently evicted** (``deque`` with ``maxlen``
    handles this transparently).  No eviction notification is emitted (v2).

    ``get_at_offset(heartbeat_index)`` retrieves an entry by its
    ``heartbeat_index`` in O(1) time by computing the offset from the oldest
    surviving entry.
    """

    def __init__(self, capacity: int | None = None) -> None:
        """Initialise the ring buffer.

        Args:
            capacity: Maximum number of entries to store.  Defaults to
                ``MemorySkillConfig.saw_buffer_capacity`` (1000).
        """
        if capacity is None:
            capacity = MemorySkillConfig.saw_buffer_capacity
        self._buffer: deque[SawEntry] = deque(maxlen=capacity)

    # ── public API ────────────────────────────────────────────────────────

    def put(self, entry: SawEntry) -> None:
        """Append a ``SawEntry`` to the buffer.

        O(1) — ``deque.append()`` with ``maxlen`` transparently evicts
        the oldest entry when the buffer is full.
        """
        self._buffer.append(entry)

    def get_all(self) -> list[SawEntry]:
        """Return all entries currently in the buffer, oldest first.

        Returns a **new list** — mutations do not affect the internal buffer.
        """
        return list(self._buffer)

    def get_at_offset(self, heartbeat_index: int) -> SawEntry | None:
        """Return the entry with the given ``heartbeat_index``, or ``None``.

        O(1) — computes the offset from the oldest surviving entry's
        ``heartbeat_index`` and indexes directly into the deque.

        Args:
            heartbeat_index: The target ``heartbeat_index`` to look up.

        Returns:
            The matching ``SawEntry`` if present, or ``None`` if the entry
            was evicted (or never inserted).
        """
        if not self._buffer:
            return None

        first_idx = self._buffer[0].heartbeat_index
        offset = heartbeat_index - first_idx
        if 0 <= offset < len(self._buffer):
            return self._buffer[offset]
        return None
