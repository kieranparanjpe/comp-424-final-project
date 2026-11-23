"""Student agent: edit ONLY this file."""
import math
import random
from typing import Any

from agents.agent import Agent
from store import register_agent
from helpers import (Move, get_valid_moves, execute_move, check_winner,
                     check_endgame, random_move, copy_state, top_view, state_key)
from constants import *
import numpy as np
import time

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
    execute_move(b, r, player, move)
    return b, r


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

    def uct(self, c=math.sqrt(2)):
        if self.num <= 0:
            return float('inf')

        if self.parent is None:
            return 1

        return self.wins / self.num + c * math.sqrt(math.log(self.parent.num) / self.num)

    def value(self):
        return self.wins / self.num if self.num > 0 else 0

    def update_score(self, winner):
        self.num += 1
        if winner == self.opponent:
            self.wins += 1
        if winner == 0:
            self.wins += 0.5

def uct_policy(node: Node) -> Move:
    return max(node.child_by_move.keys(), key=lambda mv: node.children[node.child_by_move[mv]].uct())


def check_timeout(start, margin):
    if time.perf_counter() - start >= STEP_TIME_LIMIT - margin:
        raise Timeout()


def cheap_endgame(board, reserves, player, ply):
    """Like check_endgame but only checks for a line and the ply cap; it never generates moves.

    Returns: (is_over, winner, reason), the same shape as check_endgame.
    """
    winner = check_winner(board, 3 - player)
    if winner != 0:
        return True, winner, "line"
    if ply >= MAX_PLIES:
        return True, 0, "ply_cap"
    return False, 0, ""


def score(board, reserves, player, ply):
    is_over, winner, _ = cheap_endgame(board, reserves, player, ply)
    if is_over:
        return winner

    return None

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
            node.update_score(node.opponent)
            return node.opponent

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
            while (winner := score(next_b, next_r, player, sim_ply)) is None:
                check_timeout(self.start, self.margin)
                sim_move = random_move(next_b, next_r, player)
                if sim_move is None:
                    winner = opponent
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
            return random_move(board, reserves, player)

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