"""Shared move-ordering heuristics for either backend."""

from agents.game_state import COORD_TO_INDEX, PLACEMENT_SRC, EMPTY

MAX_KILLER_PLY = 48
MAX_HISTORY_SCORE = 1_000_000
ORDER_TT_MOVE = 50_000_000
ORDER_WIN_THREAT = 20_000_000
ORDER_KILLER = 5_000_000
ORDER_BLOCK_THREAT = 3_000_000
ORDER_CAPTURE_BASE = 4000
ORDER_CAPTURE_TARGET_SIZE = 500
ORDER_CAPTURE_SELF_SIZE_PENALTY = 40
ORDER_EMPTY_BASE = 300
ORDER_EMPTY_SIZE_BONUS = 20
ORDER_SELF_STACK = 40
ORDER_DEST_HEAT_BONUS = 35
ORDER_SRC_HEAT_PENALTY = 20
ORDER_PLACEMENT_BONUS = 80


class MoveOrdering:
    def __init__(self):
        self.killers = [[None, None] for _ in range(MAX_KILLER_PLY)]
        self.history = [
            [[[0] * 3 for _ in range(16)] for _ in range(17)]
            for _ in range(3)
        ]

    def order(self, state, player, moves, tt_move, ply, my_targets=None):
        opponent = 3 - player
        my_targets = set(state.threats(player) if my_targets is None else my_targets)
        opponent_targets = set(state.threats(opponent))
        killer = self.killers[ply] if ply < len(self.killers) else (None, None)
        scored = []

        for move in moves:
            src = PLACEMENT_SRC if move.src is None else COORD_TO_INDEX[move.src]
            dest = COORD_TO_INDEX[move.dest]
            size = move.size
            score = 0
            if move == tt_move:
                score += ORDER_TT_MOVE
            if dest in my_targets:
                score += ORDER_WIN_THREAT
            if move == killer[0] or move == killer[1]:
                score += ORDER_KILLER
            if dest in opponent_targets:
                score += ORDER_BLOCK_THREAT

            dest_owner = state.own[dest]
            if dest_owner == opponent:
                score += ORDER_CAPTURE_BASE + state.sz[dest] * ORDER_CAPTURE_TARGET_SIZE - size * ORDER_CAPTURE_SELF_SIZE_PENALTY
            elif dest_owner == EMPTY:
                score += ORDER_EMPTY_BASE + size * ORDER_EMPTY_SIZE_BONUS
            else:
                score += ORDER_SELF_STACK

            score += state.heat[dest] * ORDER_DEST_HEAT_BONUS
            score += -state.heat[src] * ORDER_SRC_HEAT_PENALTY if src != PLACEMENT_SRC else ORDER_PLACEMENT_BONUS
            score += min(self.history[player][src + 1][dest][size], MAX_HISTORY_SCORE)
            scored.append((score, move))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [move for _, move in scored]

    def record_killer(self, ply, move):
        if ply < len(self.killers) and self.killers[ply][0] != move:
            self.killers[ply][1] = self.killers[ply][0]
            self.killers[ply][0] = move

    def record_history(self, player, move, depth):
        src = PLACEMENT_SRC if move.src is None else COORD_TO_INDEX[move.src]
        dest = COORD_TO_INDEX[move.dest]
        row = self.history[player][src + 1][dest]
        row[move.size] = min(row[move.size] + depth * depth, MAX_HISTORY_SCORE)
