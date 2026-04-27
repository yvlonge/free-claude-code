"""Provider-agnostic weighted target pool scheduling with cooldown tracking."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

TimeSource = Callable[[], float]


@dataclass(frozen=True, slots=True)
class WeightedTargetSlot:
    """One schedulable target identity and weight."""

    target_ref: str
    weight: int


@dataclass(frozen=True, slots=True)
class ScheduledTarget:
    """Selection result containing remaining pool health metadata."""

    target_ref: str
    retry_after_seconds: int


@dataclass(frozen=True, slots=True)
class CooldownSnapshot:
    """Cooldown timing metadata used by service error responses."""

    remaining_seconds: int


class WeightedTargetScheduler:
    """Weighted round-robin scheduler with temporary cooldown exclusion."""

    def __init__(
        self,
        slots: tuple[WeightedTargetSlot, ...],
        *,
        time_source: TimeSource | None = None,
    ) -> None:
        if not slots:
            raise ValueError("WeightedTargetScheduler requires at least one target")
        self._slots = slots
        self._clock = time_source or _default_time
        self._expanded_cycle = _expanded_cycle(slots)
        self._cursor = 0
        self._cooldowns: dict[str, float] = {}

    def next_target(self) -> ScheduledTarget | None:
        """Return the next healthy target or ``None`` when all are cooling down."""
        total = len(self._expanded_cycle)
        if total == 0:
            return None
        now = self._clock()
        cooldown = self.cooldown_snapshot(now=now)

        for _ in range(total):
            index = self._cursor % total
            self._cursor = (self._cursor + 1) % total
            target_ref = self._expanded_cycle[index]
            if _is_healthy(target_ref, self._cooldowns, now):
                return ScheduledTarget(
                    target_ref=target_ref,
                    retry_after_seconds=(
                        cooldown.remaining_seconds if cooldown is not None else 0
                    ),
                )
        return None

    def mark_unhealthy(self, target_ref: str, *, cooldown_seconds: float) -> None:
        """Place ``target_ref`` on cooldown for ``cooldown_seconds``."""
        if cooldown_seconds <= 0:
            return
        self._cooldowns[target_ref] = self._clock() + cooldown_seconds

    def cooldown_snapshot(self, *, now: float | None = None) -> CooldownSnapshot | None:
        """Return shortest remaining cooldown among currently unhealthy targets."""
        current = self._clock() if now is None else now
        remaining: list[int] = []
        for ends_at in self._cooldowns.values():
            if ends_at <= current:
                continue
            seconds = int(ends_at - current)
            if ends_at - current > seconds:
                seconds += 1
            remaining.append(max(1, seconds))
        if not remaining:
            return None
        return CooldownSnapshot(remaining_seconds=min(remaining))


def _default_time() -> float:
    import time

    return time.monotonic()


def _expanded_cycle(slots: tuple[WeightedTargetSlot, ...]) -> tuple[str, ...]:
    cycle: list[str] = []
    for slot in slots:
        cycle.extend([slot.target_ref] * slot.weight)
    return tuple(cycle)


def _is_healthy(target_ref: str, cooldowns: dict[str, float], now: float) -> bool:
    ends_at = cooldowns.get(target_ref)
    if ends_at is None:
        return True
    if ends_at <= now:
        del cooldowns[target_ref]
        return True
    return False
