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
import numpy as np
import time

SEARCH_DEPTH = 4
EVAL_FN = eval_fn1 # TODO: put this in line if we submit

class State:

    def __init__(self, key, board=None, reserves=None, player=None, wins=0, num=0, untried_moves=None, tried_moves=None):
        self.key = key
        self.board = board
        self.reserves = reserves
        self.player = player
        self.opponent = 3 - player
        
        self.wins = wins
        self.num = num

        self.untried_moves = untried_moves
        self.tried_moves = tried_moves

    def update_value_function(self, winner):
        self.num += 1
        self.wins += winner[self.opponent]

@register_agent("GraphMCTSObject")
class GraphMCTSObject(Agent):

    def __init__(self):
        """Set up your agent. This runs once per game, before the first move, and must finish
        within INIT_TIME_LIMIT seconds."""
        super().__init__()
        self.name = "GraphMCTSObject"
        self.margin = 0.05
        self.start = time.perf_counter()
        self.ply = None
        self.rng = np.random.default_rng()

        self.state_dict : dict[bytes, State] = {}

    def uct(self, state_from, state_to, c=1.41):
        parent_num = state_from.num
        wins, num = state_to.wins, state_to.num
        return wins / num + c * math.sqrt(math.log(parent_num) / num)

    def uct_policy(self, state: State) -> Move:
        move_to_child = state.tried_moves

        return max(move_to_child.keys(), key=lambda mv: self.uct(state, self.state_dict[move_to_child[mv]]))

    def rollout(self, key, ply=0):
        state = self.state_dict[key]

        s = score(state.board, state.reserves, state.player, ply)
        if s is not None:
            state.update_value_function(s)
            return s

        check_timeout(self.start, self.margin)

        if state.untried_moves is None:
            untried = get_valid_moves(state.board, state.reserves, state.player)
            random.shuffle(untried)
            state.untried_moves = untried

        if not state.untried_moves and not state.tried_moves:
            w = {state.player: 0, 3 - state.player: 1}
            state.update_value_function(w)
            return w

        if not state.untried_moves:
            move = self.uct_policy(state)
            next_key = state.tried_moves[move]
            winner = self.rollout(next_key, ply + 1)
        else:
            move = state.untried_moves[-1]

            next_b, next_r = after(state.board, state.reserves, state.player, move)
            next_player, next_opponent = state.opponent, state.player
            next_key = state_key(next_b, next_r, next_player)

            if (next_state := self.state_dict.get(next_key)) is None:
                next_state = State(next_key, board=next_b, reserves=next_r, player=state.opponent)
                self.state_dict[next_key] = next_state

            next_b, next_r = copy_state(next_b, next_r)

            sim_ply = ply + 1
            while (winner := score(next_b, next_r, next_player, sim_ply, sim_ply - ply, SEARCH_DEPTH, EVAL_FN)) is None:
                check_timeout(self.start, self.margin)

                sim_move = random_move(next_b, next_r, next_player, rng=self.rng)
                if sim_move is None:
                    winner = {next_player: 0, next_opponent : 1}
                    break
                execute_move(next_b, next_r, next_player, sim_move)
                next_player, next_opponent = next_opponent, next_player
                sim_ply += 1

            next_state.update_value_function(winner)

            state.untried_moves.pop()
            if state.tried_moves is None:
                state.tried_moves = {move : next_key}
            else:
                state.tried_moves[move] = next_key

        state.update_value_function(winner)
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
        if not (state := self.state_dict.get(key)):
            state = State(key, board, reserves, player)
            self.state_dict[key] = state

        while time.perf_counter() - self.start < STEP_TIME_LIMIT - self.margin:
            try:
                self.rollout(key, ply=self.ply)
            except Timeout:
                break

        self.ply += 2

        if not state.tried_moves:
            return random_move(board, reserves, player, rng=self.rng)

        def select(mv):
            next_state = self.state_dict[state.tried_moves[mv]]
            return next_state.num, next_state.wins / next_state.num

        move = max(state.tried_moves.keys(), key=select)
        print(f'wins: {state.wins}, number {state.num}')
        return move
