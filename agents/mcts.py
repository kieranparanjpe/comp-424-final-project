"""Student agent: edit ONLY this file."""
import math
import random
from typing import Any

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


class Node:
    def __init__(self, board, reserves, player, opponent, parent=None):
        self.board = board
        self.player = player
        self.opponent = opponent
        self.reserves = reserves
        self.parent = parent
        self.children: dict[Any, Node] = {} # key -> child node
        self.child_by_move = {} # move -> key
        self.wins = 0
        self.num = 0
        self.untried_moves = None

    def uct(self, c=1.41):
        if self.num <= 0:
            return float('inf')

        if self.parent is None:
            return 1

        return self.wins / self.num + c * math.sqrt(math.log(self.parent.num) / self.num)

    def value(self):
        return self.wins / self.num if self.num > 0 else 0

    # winner is player -> to add
    def update_score(self, winner: dict[int, float]):
        self.num += 1
        self.wins += winner[self.opponent]

def uct_policy(node: Node) -> Move:
    return max(node.child_by_move.keys(), key=lambda mv: node.children[node.child_by_move[mv]].uct())


@register_agent("VanillaMCTS")
class VanillaMCTS(Agent):

    def __init__(self):
        """Set up your agent. This runs once per game, before the first move, and must finish
        within INIT_TIME_LIMIT seconds."""
        super().__init__()
        self.name = "VanillaMCTS"
        self.last_node = None
        self.current_node = None
        self.margin = 0.05
        self.start = time.perf_counter()
        self.ply = None
        self.rng = np.random.default_rng()


    def rollout(self, node, ply=0):
        s = score(node.board, node.reserves, node.player, ply)
        if s is not None:
            node.update_score(s)
            return s
        check_timeout(self.start, self.margin)

        if node.untried_moves is None:
            node.untried_moves = get_valid_moves(node.board, node.reserves, node.player)
            random.shuffle(node.untried_moves)

        if not node.untried_moves and not node.child_by_move:
            w = {node.opponent: 1, node.player: 0}
            node.update_score(w)
            return w

        if not node.untried_moves:
            move = uct_policy(node)
            next_node = node.children[node.child_by_move[move]]
            winner = self.rollout(next_node, ply + 1)
        else:
            move = node.untried_moves[-1]
            next_b, next_r = after(node.board, node.reserves, node.player, move)
            next_node = Node(next_b, next_r, node.opponent, node.player, parent=node)
            next_key = state_key(next_b, next_r, node.opponent)

            next_b, next_r = copy_state(next_b, next_r)
            player, opponent = next_node.player, next_node.opponent
            sim_ply = ply + 1
            while (winner := score(next_b, next_r, player, sim_ply, sim_ply-ply, SEARCH_DEPTH, EVAL_FN)) is None:
                check_timeout(self.start, self.margin)
                sim_move = random_move(next_b, next_r, player, rng=self.rng)
                if sim_move is None:
                    winner = {player: 0, opponent : 1}
                    break
                execute_move(next_b, next_r, player, sim_move)
                player, opponent = opponent, player
                sim_ply += 1
            next_node.update_score(winner)

            node.untried_moves.pop()
            node.children[next_key] = next_node
            node.child_by_move[move] = next_key

        node.update_score(winner)
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

        if self.last_node is None:
            self.current_node = Node(board, reserves, player, opponent)
        else:
            current_node_key = state_key(board, reserves, player)
            self.current_node = self.last_node.children.get(current_node_key, Node(board, reserves, player, opponent))
            self.current_node.parent = None

        while time.perf_counter() - self.start < STEP_TIME_LIMIT - self.margin:
            try:
                self.rollout(self.current_node, ply=self.ply)
            except Timeout:
                break

        self.ply += 2
        if not self.current_node.child_by_move:
            self.last_node = None
            return random_move(board, reserves, player, rng=self.rng)

        move = max(
            self.current_node.child_by_move.keys(),
            key=lambda mv: (
                self.current_node.children[self.current_node.child_by_move[mv]].num,
                self.current_node.children[self.current_node.child_by_move[mv]].value(),
            ),
        )
        self.last_node = self.current_node.children[self.current_node.child_by_move[move]]
        self.last_node.parent = None
        print(self.current_node.num)
        return move