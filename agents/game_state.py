"""Shared position logic and the optimized compact-board backend."""

import constants
from helpers import LINES, Move

from agents import board_storage

BOARD_SIZE = 4
NUM_CELLS = 16
NUM_SIZES = 3
PIECE_SIZES = (0, 1, 2)
NO_SIZE = -1
PLACEMENT_SRC = -1
EMPTY = 0
PLAYER_1 = 1
PLAYER_2 = 2
OBSTACLE = 3
COORDS = tuple((r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE))
COORD_TO_INDEX = {coord: i for i, coord in enumerate(COORDS)}


def setup_lines(state, obstacle):
    """Prepare shared line masks and cell heat for a position."""
    state.masks = []
    state.line_cells = []
    state.heat = [0] * NUM_CELLS
    for line in LINES[state.W]:
        indices = tuple(r * BOARD_SIZE + c for r, c in line)
        if obstacle in indices:
            continue
        mask = sum(1 << i for i in indices)
        state.masks.append(mask)
        state.line_cells.append(indices)
        for i in indices:
            state.heat[i] += 1


def threat_targets(state, player):
    targets = []
    seen = 0
    needed = state.W - 1
    player_bits = state.bb[player]
    opponent = 3 - player
    for mask in state.masks:
        if (player_bits & mask).bit_count() != needed:
            continue
        index = (mask & ~player_bits).bit_length() - 1
        bit = 1 << index
        if seen & bit:
            continue
        if state.own[index] == opponent and state.sz[index] >= 2:
            continue
        seen |= bit
        targets.append(index)
    return targets


def can_win(state, player, targets=None):
    if targets is None:
        targets = threat_targets(state, player)
    if not targets:
        return False

    for target in targets:
        target_size = state.sz[target]
        if any(state.res[player][size] > 0 and target_size < size for size in PIECE_SIZES):
            return True

    if not state.allow_moves:
        return False

    for source in state.playable:
        if state.own[source] != player:
            continue
        size = state.sz[source]
        for target in targets:
            if target == source or state.sz[target] >= size:
                continue
            move = Move(COORDS[source], COORDS[target], size)
            token = state.apply(player, move)
            try:
                if state.terminal(player) == player:
                    return True
            finally:
                state.undo(token)
    return False


class CustomBackend:
    """Optimized backend using compact lists, bitboards and incremental hashing."""

    def __init__(self, board, reserves, player):
        self.allow_moves = bool(constants.ALLOW_MOVES)
        self.W = int(constants.WIN_LENGTH)
        self.cells = [int(value) for value in board.reshape(-1)]
        self.res = reserves.astype(int).tolist()
        self.own = [EMPTY] * NUM_CELLS
        self.sz = [NO_SIZE] * NUM_CELLS
        self.bb = [0] * (OBSTACLE + 1)
        self.playable = []
        self.z = 0

        obstacle = -1
        for index in range(NUM_CELLS):
            base = index * NUM_SIZES
            s0, s1, s2 = self.cells[base:base + NUM_SIZES]
            if s0 == OBSTACLE:
                self.own[index] = OBSTACLE
                self.bb[OBSTACLE] |= 1 << index
                obstacle = index
            else:
                self.playable.append(index)
                if s2:
                    owner, size = s2, 2
                elif s1:
                    owner, size = s1, 1
                elif s0:
                    owner, size = s0, 0
                else:
                    owner, size = EMPTY, NO_SIZE
                self.own[index], self.sz[index] = owner, size
                self.bb[owner] |= 1 << index

        for index, value in enumerate(self.cells):
            self.z |= value << (index * board_storage.BITS_PER_ENTRY)
        for p in (PLAYER_1, PLAYER_2):
            for size in PIECE_SIZES:
                board_storage.write_reserve_hash(self, p, size)
        if player == PLAYER_2:
            self.z |= board_storage.PLAYER_BIT

        setup_lines(self, obstacle)

    def get_valid_moves(self, player):
        from agents.move_generation import generate_custom_moves
        return generate_custom_moves(self, player)

    def apply(self, player, move):
        src = PLACEMENT_SRC if move.src is None else COORD_TO_INDEX[move.src]
        dest = COORD_TO_INDEX[move.dest]
        size = move.size
        old_dest = (self.own[dest], self.sz[dest])
        old_source = (EMPTY, NO_SIZE)

        if src == PLACEMENT_SRC:
            self.res[player][size] -= 1
            board_storage.write_reserve_hash(self, player, size)
        else:
            old_source = (self.own[src], self.sz[src])
            source_slot = src * NUM_SIZES + size
            self.cells[source_slot] = EMPTY
            board_storage.write_slot_hash(self, source_slot, EMPTY)
            owner, top_size = EMPTY, NO_SIZE
            for lower_size in range(size - 1, -1, -1):
                value = self.cells[src * NUM_SIZES + lower_size]
                if value:
                    owner, top_size = value, lower_size
                    break
            self._set_top(src, owner, top_size)

        dest_slot = dest * NUM_SIZES + size
        self.cells[dest_slot] = player
        board_storage.write_slot_hash(self, dest_slot, player)
        self._set_top(dest, player, size)
        self.z ^= board_storage.PLAYER_BIT
        return src, dest, size, player, old_dest, old_source

    def undo(self, token):
        src, dest, size, player, old_dest, old_source = token
        dest_slot = dest * NUM_SIZES + size
        self.cells[dest_slot] = EMPTY
        board_storage.write_slot_hash(self, dest_slot, EMPTY)
        self._set_top(dest, *old_dest)

        if src == PLACEMENT_SRC:
            self.res[player][size] += 1
            board_storage.write_reserve_hash(self, player, size)
        else:
            source_slot = src * NUM_SIZES + size
            self.cells[source_slot] = player
            board_storage.write_slot_hash(self, source_slot, player)
            self._set_top(src, *old_source)
        self.z ^= board_storage.PLAYER_BIT

    def _set_top(self, index, owner, size):
        old_owner = self.own[index]
        if old_owner != owner:
            bit = 1 << index
            self.bb[old_owner] &= ~bit
            self.bb[owner] |= bit
            self.own[index] = owner
        self.sz[index] = size

    def terminal(self, mover):
        for player in (3 - mover, mover):
            bits = self.bb[player]
            if any((bits & mask) == mask for mask in self.masks):
                return player
        return EMPTY

    def key(self, player):
        return self.z

    def threats(self, player):
        return threat_targets(self, player)

    def can_win(self, player, targets=None):
        return can_win(self, player, targets)
