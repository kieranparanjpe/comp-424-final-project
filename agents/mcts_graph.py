"""Student agent: edit ONLY this file."""
import math
import random

# TODO: remove
from agents.evaluation_functions import eval_fn1
from agents.mcts_helpers import Timeout, after, check_timeout, score

from agents.agent import Agent

from store import register_agent
from helpers import (Move, get_valid_moves, execute_move, check_winner,
                     check_endgame, random_move, copy_state, top_view, state_key)
from constants import *
import constants
import numpy as np
import time

SEARCH_DEPTH = 4
EVAL_FN = eval_fn1 # TODO: put this in line if we submit

@register_agent("GraphMCTS")
class GraphMCTS(Agent):

    def __init__(self):
        """Set up your agent. This runs once per game, before the first move, and must finish
        within INIT_TIME_LIMIT seconds."""
        super().__init__()
        self.name = "GraphMCTS"
        self.margin = 0.05
        self.start = time.perf_counter()
        self.ply = None
        self.rng = np.random.default_rng()
        self.value_function : dict[bytes, tuple[int, int]] = {} # key -> (wins, num)
        self.untried_moves : dict[bytes, list[Move] | None] = {}
        self.tried_moves : dict[bytes, dict[Move, bytes] | None] = {} # key -> Move -> key
        self.states: dict[bytes, tuple] = {}

    def uct(self, key_from, key_to, c=1.41):
        _, parent_num = self.value_function[key_from]
        wins, num = self.value_function[key_to]
        return wins / num + c * math.sqrt(math.log(parent_num) / num)

    def uct_policy(self, key) -> Move:
        move_to_child = self.tried_moves[key]

        return max(move_to_child.keys(), key=lambda mv: self.uct(key, move_to_child[mv]))

    def update_value_function(self, key, winner, player):
        opponent = 3 - player
        if (win_num := self.value_function.get(key, (0, 0))) is not None:
            self.value_function[key] = (win_num[0] + winner[opponent], win_num[1] + 1)

    def rollout(self, board, reserves, player, ply=0):
        s = score(board, reserves, player, ply)
        key = state_key(board, reserves, player)
        if s is not None:
            self.update_value_function(key, s, player)
            return s
        check_timeout(self.start, self.margin)

        if self.untried_moves.get(key) is None:
            untried = get_valid_moves(board, reserves, player) or ([None] if not constants.ALLOW_MOVES else [])
            random.shuffle(untried)
            self.untried_moves[key] = untried

        if not self.untried_moves.get(key) and not self.tried_moves.get(key):
            w = {player: 0, 3 - player: 1}
            self.update_value_function(key, w, player)
            return w

        if not self.untried_moves.get(key):
            move = self.uct_policy(key)
            next_key = self.tried_moves[key][move]
            nxt = self.states[next_key]
            winner = self.rollout(*nxt, ply + 1)
        else:
            move = self.untried_moves[key][-1]

            next_b, next_r = after(board, reserves, player, move)
            next_player, next_opponent = 3 - player, player
            next_key = state_key(next_b, next_r, next_player)

            self.states[next_key] = (next_b, next_r, 3 - player)
            next_b, next_r = copy_state(next_b, next_r)

            sim_ply = ply + 1
            while (winner := score(next_b, next_r, next_player, sim_ply, sim_ply - ply, SEARCH_DEPTH, EVAL_FN)) is None:
                check_timeout(self.start, self.margin)

                sim_move = random_move(next_b, next_r, next_player, rng=self.rng)
                if sim_move is None:
                    if not constants.ALLOW_MOVES:
                        next_player, next_opponent = next_opponent, next_player
                        sim_ply += 1
                        continue
                    winner = {next_player: 0, next_opponent : 1}
                    break
                execute_move(next_b, next_r, next_player, sim_move)
                next_player, next_opponent = next_opponent, next_player
                sim_ply += 1

            self.update_value_function(next_key, winner, 3 - player)

            self.untried_moves[key].pop()
            if not key in self.tried_moves:
                self.tried_moves[key] = {move : next_key}
            else:
                self.tried_moves[key][move] = next_key

        self.update_value_function(key, winner, player)
        return winner


    def step(self, board, reserves, player, opponent):
        """Return the Move you want to play.

        Takes: board, reserves - your own copies of the current position.
               player, opponent - your player number (1 or 2) and the other one.
        Agent.step in agents/agent.py explains each argument and the time limit.
        """
        self.start = time.perf_counter()
        # ALLOW_MOVES and WIN_LENGTH (from constants) are the rule flags of this game.
        # helpers.get_valid_moves / check_winner already respect them; your own
        # move generator and evaluation function must too.
        # TODO: replace with your search. Keep (time.perf_counter() - start) < STEP_TIME_LIMIT - margin.
        if self.ply is None:
            self.ply = 0 if player == 1 else 1

        key = state_key(board, reserves, player)
        self.states[key] = (board, reserves, player)

        num_rollouts = 0
        while time.perf_counter() - self.start < STEP_TIME_LIMIT - self.margin:
            try:
                num_rollouts += 1
                self.rollout(board, reserves, player, ply=self.ply)
            except Timeout:
                break

        self.ply += 2

        if not self.tried_moves.get(key):
            return random_move(board, reserves, player, rng=self.rng)

        def select(mv):
            w, n = self.value_function[self.tried_moves[key][mv]]
            return n, w / n

        move = max(self.tried_moves[key].keys(), key=select)
        print(f"Simulated games: {num_rollouts}")
        return move
