"""Custom legal-move generation for the optimized backend."""

from helpers import Move

from agents.game_state import COORDS, PIECE_SIZES


def generate_custom_moves(state, player):
    moves = [
        Move(None, COORDS[dest], size)
        for size in PIECE_SIZES if state.res[player][size] > 0
        for dest in state.playable if state.sz[dest] < size
    ]
    if state.allow_moves:
        for source in state.playable:
            if state.own[source] != player:
                continue
            size = state.sz[source]
            moves.extend(
                Move(COORDS[source], COORDS[dest], size)
                for dest in state.playable
                if dest != source and state.sz[dest] < size
            )
    return moves
