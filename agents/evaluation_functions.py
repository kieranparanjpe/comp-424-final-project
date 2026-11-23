import numpy as np

from helpers import top_view, _LINE_ROWS, _LINE_COLS

def four_line_counts(board, player, opponent):
    owners, _ = top_view(board)
    k = 4
    cells = owners[_LINE_ROWS[k], _LINE_COLS[k]]  # (10, 4)

    mine = (cells == player).sum(axis=1)  # how many pieces do i have in each 4-line. (10,)
    theirs = (cells == opponent).sum(axis=1) # how many pieces do they have in each 4-line. (10,)
    blocked = (cells == 3).any(axis=1)  # does each 4-line have the obstacle? (10,)

    open_lines = ~blocked & (theirs == 0)  # find 4 lines that have no obstacle and have no opponent pieces
    counts = np.bincount(mine[open_lines], minlength=k + 1) # give a histogram where counts[k] is the number of open 4-lines that i have
    # k-full
    return counts[2:5]

weights_1 = np.array([1, 2, 3])
best_score = weights_1[2] * 1 + weights_1[1] * 2

# returns dict giving {player: score, opponent: score}
def eval_fn1(board, player, opponent):
    p = four_line_counts(board, player, opponent)
    o = four_line_counts(board, opponent, player)

    score1 = np.dot(p, weights_1)
    score2 = np.dot(o, weights_1)
    p1_score = 0.5 * ((score1 - score2) / best_score + 1)

    return {player : p1_score, opponent: 1 - p1_score}

