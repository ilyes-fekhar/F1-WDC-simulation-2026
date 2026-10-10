from clinch import (
    earliest_clinch,
    first_decided_round,
    is_decided,
    max_remaining,
    still_possible,
)

# After round 16 of 2026 (Bahrain). Singapore is the last sprint.
STANDINGS_2026 = {
    "ANT": 320, "RUS": 236, "HAM": 214, "LEC": 191,
    "NOR": 188, "VER": 188, "PIA": 128, "HAD": 96,
}
ROUNDS_LEFT_2026 = [True] + [False] * 6  # SIN (sprint), USA, MEX, BRA, LAS, QAT, ABU


def test_max_remaining():
    assert max_remaining(ROUNDS_LEFT_2026) == 183
    assert max_remaining([]) == 0


def test_not_decided_now():
    assert not is_decided(STANDINGS_2026, ROUNDS_LEFT_2026)


def test_decided_when_lead_exceeds_remaining():
    assert is_decided({"A": 300, "B": 270}, [False])  # lead 30 > 25


def test_tie_is_not_decided():
    assert not is_decided({"A": 295, "B": 270}, [False])  # lead 25 == 25


def test_still_possible_2026():
    alive = set(still_possible(STANDINGS_2026, ROUNDS_LEFT_2026))
    assert alive == {"ANT", "RUS", "HAM", "LEC", "NOR", "VER"}  # PIA: 128+183 = 311 < 320


def test_earliest_clinch_is_round_18():
    # index 0 = Singapore (R17), index 1 = USA (R18)
    assert earliest_clinch(STANDINGS_2026, ROUNDS_LEFT_2026, "ANT") == 1


def test_realistic_path_clinches_round_19():
    # ANT wins everything, RUS finishes P2 every time (sprint P2 = 7, GP P2 = 18)
    round_points = [{"ANT": 33, "RUS": 25}] + [{"ANT": 25, "RUS": 18}] * 6
    assert first_decided_round(STANDINGS_2026, ROUNDS_LEFT_2026, round_points) == 2


def test_comeback_path_decides_round_22():
    # RUS wins every session, ANT scores nothing: RUS takes the lead and the
    # title is decided at Qatar (index 5), not Abu Dhabi.
    round_points = [{"RUS": 33}] + [{"RUS": 25}] * 6
    assert first_decided_round(STANDINGS_2026, ROUNDS_LEFT_2026, round_points) == 5


def test_fastest_lap_point_counts_in_remaining():
    # 2023 after Japan: 6 rounds left, 3 sprints. F1's own figure was 180, not 174.
    left = [True, True, False, True, False, False]
    assert max_remaining(left) == 174
    assert max_remaining(left, fl=1) == 180


def test_2023_lead_of_177_was_not_decided_after_japan():
    st = {"ver": 400, "per": 223}
    left = [True, True, False, True, False, False]
    assert is_decided(st, left)                 # without the bonus point: wrongly decided
    assert not is_decided(st, left, fl=1)       # with it: not decided, as in reality
