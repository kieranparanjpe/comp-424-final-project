"""Shared evaluation function used by both backends."""

from agents.game_state import PIECE_SIZES, NO_SIZE

MATE = 100_000
DOUBLE_THREAT_MIN = 2
TOP_PIECE_VALUES = (16, 34, 58)
BURIED_PIECE_VALUES = (3, 6, 10)
RESERVE_PIECE_VALUES = (6, 16, 40)
OPEN_LINE_SCORES = {1: 4800, 2: 170, 3: 26}
EVAL_HEAT_BASE = 4
EVAL_MULTI_THREAT_PENALTY = 22_000
EVAL_DOUBLE_THREAT_BONUS = 2800
EVAL_SCORE_CLAMP = 80_000
LINE_OPEN_SIZE_WEIGHT = 3
LINE_BLOCKED_THREAT_BASE = 900
LINE_BLOCKED_THREAT_SIZE_WEIGHT = 2
LINE_BLOCKED_SETUP_BASE = 60
LINE_CAPPED_WEIGHT = 5
LINE_CONTESTED_WEIGHT = 4
LARGE_SIZE = 2


def open_points(count, win_length):
    return OPEN_LINE_SCORES.get(win_length - count, 0)


def evaluate(state, player):
    my_targets = state.threats(player)
    if state.can_win(player, my_targets):
        return MATE - 1
    opponent = 3 - player
    opponent_targets = state.threats(opponent)
    score = material(state, player) - material(state, opponent)
    score += lines(state, player) - lines(state, opponent)
    if len(opponent_targets) >= DOUBLE_THREAT_MIN:
        score -= EVAL_MULTI_THREAT_PENALTY
    return max(-EVAL_SCORE_CLAMP, min(EVAL_SCORE_CLAMP, score))


def material(state, player):
    score = 0
    for index in state.playable:
        top_size, owner, heat = state.sz[index], state.own[index], state.heat[index]
        if owner != player:
            continue
        base = index * 3
        for size in PIECE_SIZES:
            if state.cells[base + size] == player:
                if size == top_size:
                    score += TOP_PIECE_VALUES[size] + heat * (EVAL_HEAT_BASE + size)
                else:
                    score += BURIED_PIECE_VALUES[size]
    return score + sum(state.res[player][size] * RESERVE_PIECE_VALUES[size] for size in PIECE_SIZES)


def lines(state, player):
    opponent = 3 - player
    total = threats = 0
    for line_cells in state.line_cells:
        own_count = opponent_count = size_sum = 0
        opponent_size = NO_SIZE
        for index in line_cells:
            owner = state.own[index]
            if owner == player:
                own_count += 1
                size_sum += state.sz[index] + 1
            elif owner == opponent:
                opponent_count += 1
                opponent_size = state.sz[index]
        if own_count == 0:
            continue
        if opponent_count == 0:
            total += open_points(own_count, state.W) + size_sum * LINE_OPEN_SIZE_WEIGHT
            if own_count == state.W - 1:
                threats += 1
        elif opponent_count == 1 and opponent_size < LARGE_SIZE and own_count == state.W - 1:
            total += LINE_BLOCKED_THREAT_BASE + size_sum * LINE_BLOCKED_THREAT_SIZE_WEIGHT
            threats += 1
        elif opponent_count == 1 and opponent_size < LARGE_SIZE and own_count == state.W - 2:
            total += LINE_BLOCKED_SETUP_BASE + size_sum
        elif opponent_count == 1 and opponent_size >= LARGE_SIZE:
            total += LINE_CAPPED_WEIGHT * own_count
        elif own_count > opponent_count:
            total += LINE_CONTESTED_WEIGHT * own_count
    if threats >= DOUBLE_THREAT_MIN:
        total += EVAL_DOUBLE_THREAT_BONUS
    return total
