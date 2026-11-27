"""Student agent: shared iterative-deepening negamax alpha-beta search."""

import time

import constants
from agents.agent import Agent
from agents.game_state import CustomBackend
from agents.original_backend import OriginalBackend
from agents.move_ordering import MoveOrdering
from helpers import check_move_validity, get_valid_moves
from store import register_agent
from agents import evaluation

# False uses the framework helpers directly; True uses the optimized backend.
USE_CUSTOM_HELPERS = True

DEFAULT_MAX_PLIES = 100
TIME_MARGIN = 0.25
SOFT_TIME_RATIO = 0.45
MIN_SEARCH_BUDGET = 0.05
MAX_SEARCH_DEPTH = 32
MAX_TT_ENTRIES = 250_000
TIME_CHECK_MASK = 63
ENDGAME_TT_PLY_THRESHOLD = 14
IMMEDIATE_WIN_DEPTH = 64
MATE = 100_000
MATE_TH = 90_000
INF = 1_000_000
EXACT, LOWER, UPPER = 0, 1, 2


class _TimeUp(Exception):
    pass


@register_agent("student_agent")
class StudentAgent(Agent):
    """Runs one search implementation over either of the two game backends."""

    def __init__(self):
        super().__init__()
        self.name = "StudentAgent"
        self.use_custom_helpers = USE_CUSTOM_HELPERS
        self.ordering = MoveOrdering()
        self.last_depth = 0
        self.last_score = 0
        self.last_nodes = 0
        self.turn_count = 0
        self.tt = {}
        self._tt_backend = None

    def step(self, board, reserves, player, opponent):
        start = time.perf_counter()
        current_ply = 2 * self.turn_count + (int(player) - 1)
        self.turn_count += 1
        max_plies = int(getattr(constants, "MAX_PLIES", DEFAULT_MAX_PLIES))
        move = self._choose(board, reserves, player, start, max(0, max_plies - current_ply))

        elapsed = time.perf_counter() - start

        print(
            f"Backend: {type(self.state).__name__} | "
            f"Time: {elapsed:.4f}s | "
            f"Nodes: {self.last_nodes} | "
            f"Depth: {self.last_depth}"
        )

        if move is not None and check_move_validity(board, reserves, player, move):
            return move
        legal_moves = get_valid_moves(board, reserves, player)
        if not legal_moves:
            raise RuntimeError("step() was called when the player has no legal moves")

        return legal_moves[0]

    def _choose(self, board, reserves, player, start, plies_left=DEFAULT_MAX_PLIES):
        if self.use_custom_helpers:
            self.state = CustomBackend(board, reserves, player)
            backend_name = "custom"
        else:
            self.state = OriginalBackend(board, reserves, player)
            backend_name = "original"
        signature = (backend_name, int(constants.WIN_LENGTH), bool(constants.ALLOW_MOVES))
        if self._tt_backend != signature or len(self.tt) > MAX_TT_ENTRIES:
            self.tt.clear()
            self._tt_backend = signature

        self.nodes = 0
        self.plies_left = max(0, int(plies_left))
        budget = max(MIN_SEARCH_BUDGET, constants.STEP_TIME_LIMIT - TIME_MARGIN)
        self.deadline = start + budget
        soft_deadline = start + budget * SOFT_TIME_RATIO
        moves = self.state.get_valid_moves(player)
        if not moves:
            raise RuntimeError("no legal moves")

        ordered = self.ordering.order(self.state, player, moves, None, 0)
        best = ordered[0]
        self.last_depth = 0

        for depth in range(1, min(MAX_SEARCH_DEPTH, self.plies_left + 1)):
            now = time.perf_counter()
            if now >= self.deadline or (depth > 1 and now >= soft_deadline):
                break
            self._root_partial = None
            try:
                score, move = self._search_root(depth, player, ordered)
            except _TimeUp:
                if self._root_partial is not None:
                    score, move = self._root_partial
                    if move is not None:
                        best = move
                        self.last_score = score
                break
            if move is not None:
                best = move
                self.last_depth = depth
                self.last_score = score
            if score is not None and abs(score) > MATE_TH:
                break
            ordered = self.ordering.order(self.state, player, moves, best, 0)

        self.last_nodes = self.nodes
        return best

    @staticmethod
    def _adjust(value):
        if value > MATE_TH:
            return value - 1
        if value < -MATE_TH:
            return value + 1
        return value

    def _tt_key(self, player, plies_left):
        key = self.state.key(player)
        return key if plies_left > ENDGAME_TT_PLY_THRESHOLD else (key, plies_left)

    def _search_root(self, depth, player, moves):
        alpha, beta = -INF, INF
        alpha0 = alpha
        best, best_move = -INF, None
        opponent = 3 - player
        key = self._tt_key(player, self.plies_left)

        for move in moves:
            if time.perf_counter() >= self.deadline:
                raise _TimeUp
            token = self.state.apply(player, move)
            won = revealed = False
            value = 0
            try:
                winner = self.state.terminal(player)
                if winner == player:
                    won = True
                elif winner == opponent:
                    revealed = True
                elif self.plies_left <= 1:
                    value = 0
                else:
                    value = self._adjust(
                        -self._negamax(depth - 1, -beta, -alpha, opponent, 1, self.plies_left - 1)
                    )
            finally:
                self.state.undo(token)

            if won:
                self.tt[key] = (depth, MATE - 1, EXACT, move)
                return MATE - 1, move
            if revealed:
                continue
            if value > best:
                best, best_move = value, move
            if value > alpha:
                alpha = value
            self._root_partial = (best, best_move)

        if best_move is None:
            self.tt[key] = (depth, -(MATE - 1), EXACT, moves[0])
            return -(MATE - 1), moves[0]
        flag = UPPER if best <= alpha0 else (LOWER if best >= beta else EXACT)
        self.tt[key] = (depth, best, flag, best_move)
        return best, best_move

    def _negamax(self, depth, alpha, beta, player, ply, plies_left):
        self.nodes += 1
        if (self.nodes & TIME_CHECK_MASK) == 0 and time.perf_counter() >= self.deadline:
            raise _TimeUp
        if plies_left <= 0:
            return 0

        alpha0 = alpha
        key = self._tt_key(player, plies_left)
        entry = self.tt.get(key)
        tt_move = None
        if entry is not None:
            tt_depth, tt_score, tt_flag, tt_move = entry
            if tt_depth >= depth:
                if tt_flag == EXACT or (tt_flag == LOWER and tt_score >= beta) or (tt_flag == UPPER and tt_score <= alpha):
                    return tt_score

        if depth <= 0:
            return evaluation.evaluate(self.state, player)

        my_targets = self.state.threats(player)
        if self.state.can_win(player, my_targets):
            self.tt[key] = (IMMEDIATE_WIN_DEPTH, MATE - 1, EXACT, tt_move)
            return MATE - 1

        moves = self.state.get_valid_moves(player)
        if not moves:
            return -(MATE - 1)
        moves = self.ordering.order(self.state, player, moves, tt_move, ply, my_targets)

        best, best_move = -INF, None
        opponent = 3 - player
        for move in moves:
            token = self.state.apply(player, move)
            won = revealed = False
            value = 0
            try:
                winner = self.state.terminal(player)
                if winner == player:
                    won = True
                elif winner == opponent:
                    revealed = True
                elif plies_left <= 1:
                    value = 0
                else:
                    value = self._adjust(
                        -self._negamax(depth - 1, -beta, -alpha, opponent, ply + 1, plies_left - 1)
                    )
            finally:
                self.state.undo(token)

            if won:
                self.ordering.record_killer(ply, move)
                self.tt[key] = (depth, MATE - 1, EXACT, move)
                return MATE - 1
            if revealed:
                continue
            if value > best:
                best, best_move = value, move
            if value > alpha:
                alpha = value
            if alpha >= beta:
                self.ordering.record_killer(ply, move)
                self.ordering.record_history(player, move, depth)
                break

        if best_move is None:
            score = -(MATE - 1)
            self.tt[key] = (depth, score, EXACT, moves[0])
            return score
        flag = UPPER if best <= alpha0 else (LOWER if best >= beta else EXACT)
        self.tt[key] = (depth, best, flag, best_move)
        return best
