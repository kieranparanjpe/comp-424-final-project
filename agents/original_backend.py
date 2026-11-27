"""Reference backend that uses the framework's helpers directly."""

import constants
from helpers import (
    LINES,
    check_winner,
    copy_state,
    execute_move,
    get_valid_moves,
    state_key,
    top_view,
)

from agents.game_state import (
    BOARD_SIZE,
    NUM_CELLS,
    NUM_SIZES,
    EMPTY,
    OBSTACLE,
    setup_lines,
    threat_targets,
    can_win,
)


class OriginalBackend:
    """Uses NumPy board states and calls the original helpers for game operations."""

    def __init__(self, board, reserves, player):
        self.board = board.copy()
        self.reserves = reserves.copy()
        self.allow_moves = bool(constants.ALLOW_MOVES)
        self.W = int(constants.WIN_LENGTH)
        self._refresh()

    def _refresh(self):
        owners, sizes = top_view(self.board)
        self.own = owners.reshape(-1).astype(int).tolist()
        self.sz = sizes.reshape(-1).astype(int).tolist()
        self.cells = self.board.reshape(-1).astype(int).tolist()
        self.res = self.reserves.astype(int).tolist()
        self.playable = [i for i, owner in enumerate(self.own) if owner != OBSTACLE]
        self.bb = [0] * (OBSTACLE + 1)
        obstacle = -1
        for index, owner in enumerate(self.own):
            self.bb[owner] |= 1 << index
            if owner == OBSTACLE:
                obstacle = index
        setup_lines(self, obstacle)

    def get_valid_moves(self, player):
        # This is the framework helper itself, not a reimplementation or wrapper.
        return get_valid_moves(self.board, self.reserves, player, self.allow_moves)

    def apply(self, player, move):
        old_board, old_reserves = self.board, self.reserves
        self.board, self.reserves = copy_state(self.board, self.reserves)
        execute_move(self.board, self.reserves, player, move, self.allow_moves)
        self._refresh()
        return old_board, old_reserves

    def undo(self, token):
        self.board, self.reserves = token
        self._refresh()

    def terminal(self, mover):
        return check_winner(self.board, mover, self.W)

    def key(self, player):
        return state_key(self.board, self.reserves, player)

    def threats(self, player):
        return threat_targets(self, player)

    def can_win(self, player, targets=None):
        return can_win(self, player, targets)
