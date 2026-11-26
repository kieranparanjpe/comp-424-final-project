import time

import numpy as np

import constants
from helpers import check_winner, copy_state, execute_move
from constants import *


class Timeout(Exception):
    pass


def after(board, reserves, player, move):
    """Return the position after `player` plays `move`. The original arrays are not changed.

    Takes:   board, reserves - the position before the move.
             player          - the player who moves, 1 or 2.
             move            - a legal Move for that player.
    Returns: (board, reserves), new arrays with the move played.
    Example: b2, r2 = after(board, reserves, 1, Move(None, (0, 0), LARGE))
    """
    b, r = copy_state(board, reserves)
    if move is not None:  # None is a pass
        execute_move(b, r, player, move)
    return b, r


def check_timeout(start, margin):
    if time.perf_counter() - start >= STEP_TIME_LIMIT - margin:
        raise Timeout()


def can_place(board, reserves, player):
    have = np.nonzero(reserves[player])[0]
    if have.size == 0:
        return False

    return bool((board[:, :, have[-1]:] == 0).all(axis=2).any())


def cheap_endgame(board, reserves, player, ply):
    winner = check_winner(board, 3 - player)
    if winner != 0:
        return True, winner, "line"
    if ply >= MAX_PLIES:
        return True, 0, "ply_cap"
    if not constants.ALLOW_MOVES and not can_place(board, reserves, player) and not can_place(board, reserves, 3 - player):
        return True, 0, "exhausted"
    return False, 0, ""


def score(board, reserves, player, ply, relative_depth=None, maximum_depth=None, eval_fn=None):
    is_over, winner, _ = cheap_endgame(board, reserves, player, ply)
    if is_over:
        if winner == 0: return {player: 0.5, 3 - player: 0.5}
        return {winner: 1, 3 - winner: 0}

    if relative_depth is not None and maximum_depth is not None and eval_fn is not None and relative_depth > maximum_depth:
        return eval_fn(board, player, 3 - player)

    return None
