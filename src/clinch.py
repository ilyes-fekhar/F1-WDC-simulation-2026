"""Deterministic title-clinch logic. Pure functions, no I/O.

Conventions
- standings: dict {driver: points}
- rounds_left: list of bools, one per remaining round, True = sprint weekend
- Ties are treated conservatively (strict '>'): the wins countback is ignored.
"""
from __future__ import annotations

GP_WIN = 25
SPRINT_WIN = 8


def max_remaining(rounds_left: list[bool], fl: int = 0) -> int:
    """Most points any single driver can still score.
    fl = fastest-lap bonus per GP (1 in 2019-2024, 0 otherwise)."""
    return sum(GP_WIN + fl + (SPRINT_WIN if is_sprint else 0) for is_sprint in rounds_left)


def is_decided(standings: dict[str, float], rounds_left: list[bool], fl: int = 0) -> bool:
    """True if the leader is champion whatever happens in the remaining rounds."""
    ranked = sorted(standings.values(), reverse=True)
    if len(ranked) < 2:
        return True
    return ranked[0] > ranked[1] + max_remaining(rounds_left, fl)


def still_possible(standings: dict[str, float], rounds_left: list[bool], fl: int = 0) -> list[str]:
    """Drivers who can still mathematically finish level with or above the leader."""
    leader_pts = max(standings.values())
    rem = max_remaining(rounds_left, fl)
    return [d for d, p in standings.items() if p + rem >= leader_pts]


def first_decided_round(
    standings: dict[str, float],
    rounds_left: list[bool],
    round_points: list[dict[str, float]],
    fl: int = 0,
) -> int | None:
    """Apply per-round points in order; return the 0-based index (into rounds_left)
    of the first round after which the title is decided, or None.

    round_points[i] = {driver: points gained in round i}. This is the function the
    Monte Carlo simulator will call once per simulated season.
    """
    pts = dict(standings)
    for i, gained in enumerate(round_points):
        for driver, p in gained.items():
            pts[driver] = pts.get(driver, 0) + p
        if is_decided(pts, rounds_left[i + 1:], fl):
            return i
    return None


def earliest_clinch(
    standings: dict[str, float], rounds_left: list[bool], leader: str, fl: int = 0
) -> int | None:
    """Best case for the leader: wins every session, all rivals score nothing."""
    round_points = [
        {leader: GP_WIN + fl + (SPRINT_WIN if is_sprint else 0)} for is_sprint in rounds_left
    ]
    return first_decided_round(standings, rounds_left, round_points, fl)
