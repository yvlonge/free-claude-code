from __future__ import annotations

from core.scheduling import WeightedTargetScheduler, WeightedTargetSlot


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_weighted_scheduler_rotates_by_weight() -> None:
    clock = FakeClock()
    scheduler = WeightedTargetScheduler(
        (
            WeightedTargetSlot("a", 2),
            WeightedTargetSlot("b", 1),
        ),
        time_source=clock,
    )

    picks = [scheduler.next_target() for _ in range(6)]
    assert [pick.target_ref for pick in picks if pick is not None] == [
        "a",
        "a",
        "b",
        "a",
        "a",
        "b",
    ]


def test_weighted_scheduler_skips_cooldown_targets() -> None:
    clock = FakeClock()
    scheduler = WeightedTargetScheduler(
        (
            WeightedTargetSlot("a", 1),
            WeightedTargetSlot("b", 1),
        ),
        time_source=clock,
    )

    first = scheduler.next_target()
    assert first is not None
    scheduler.mark_unhealthy(first.target_ref, cooldown_seconds=10)

    second = scheduler.next_target()
    assert second is not None
    assert second.target_ref != first.target_ref


def test_weighted_scheduler_returns_none_when_all_unhealthy() -> None:
    clock = FakeClock()
    scheduler = WeightedTargetScheduler(
        (
            WeightedTargetSlot("a", 1),
            WeightedTargetSlot("b", 1),
        ),
        time_source=clock,
    )

    scheduler.mark_unhealthy("a", cooldown_seconds=5)
    scheduler.mark_unhealthy("b", cooldown_seconds=7)

    assert scheduler.next_target() is None
    snapshot = scheduler.cooldown_snapshot()
    assert snapshot is not None
    assert snapshot.remaining_seconds == 5


def test_weighted_scheduler_recovers_after_cooldown() -> None:
    clock = FakeClock()
    scheduler = WeightedTargetScheduler(
        (WeightedTargetSlot("a", 1),),
        time_source=clock,
    )

    scheduler.mark_unhealthy("a", cooldown_seconds=3)
    assert scheduler.next_target() is None

    clock.advance(3.1)
    selected = scheduler.next_target()
    assert selected is not None
    assert selected.target_ref == "a"
