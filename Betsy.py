#!/bin/python

# I used the basic alpha beta pruning algorithm described in the textbook, however I made my own heuristic
# function to evaluate and return a value for the current state of the board. I was initially taking the top
# n rows of the board and pasting the bottom row onto the top of those top n rows, then giving each player a
# weighted score for where their pieces were, assigning a score of 4 for a piece in the top (really the bottom pasted on top)
# row and so on down to 1 for the bottom of the top n rows. I also gave a 1 for the first space in a column to the player
# whose turn it was, as that represented a potential place they could drop a piece. However, this seemed a little too
# abstracted from calculating potential wins, so I ended up looking at each row and it's neighbor and adding 1 to the value
# for any potential wins, and then doing the same for any potential wins in each column, and each diagonal plus the
# diagonal above it. In my first attempt I ended up with an infinite loop as my successor function started by
# rotating the first column, and in the back and forth between max and min functions it just kept rotating the first column
# infinitely. To solve this, I started tracking the path of moves (which also helped me return the first move in the
# series that led to a win) and stopping the iteration if it had rotated a row more than n times. I also randomized the
# order of moves in my successor function so that my opponent wouldn't be able to predict what move I would make based on
# the fact that my successors were always generated in the same order.

import sys
import numpy as np
import math
import random

n = int(sys.argv[1])
startPlayer = sys.argv[2]
startBoard = list(sys.argv[3])
time = sys.argv[4]


alpha = -float('inf')
beta = float('inf')
recommend = ''


def drop(player, column, board, n):
    board1 = list(board)
    for i in range((n * (n + 3)) - 1 - (n - column), -1, -n):
        if board1[i] == '.':
            board1[i]= player
            return board1


def rotate(column, board, n):
    board1 = list(board)
    for i in range((n * (n + 3)) - 1 - (n - column), -1, -n):
        board1[i] = board[i-n]
    for k in range(0+(column-1), (n*(n+3))-n, n):
        if board1[k] != '.' and board1[k+n] == '.':
            board1[k], board1[k+n] = board1[k+n], board1[k]
    return board1


def successors(player, board, n):
    actions = []
    board1 = list(board)
    for i in range(1, n + 1):
        a = drop(player, i, board1, n)
        if a is not None:
            actions.append(['drop', i, a])
        actions.append(['rotate', i, rotate(i, board1, n)])
    return actions

def value(player, board, n):
    value = 0
    board1 = (board[-n:] + board[0:(n * n)])
    board2 = np.reshape(board1, (n + 1, n))
    i = n
    j = 0
    while j < n and i > 0:
        if board2[i][j] == player or board2[i - 1][j] == player:
            j += 1
            if j == n:
                value += 1
                i -= 1
                j = 0
        else:
            i -= 1
            j = 0
    for k in range(0,n):
        for m in range(n, -1, -1):
            if board2[m][k] == player:
                value+=1
            elif board2[m][k] !=player and board2[m][k]!='.':
                value-=1
    diag1 = board2[1:].diagonal()
    diag11 = board2[:n].diagonal()
    flipboard = np.fliplr(board2)
    diag2 = flipboard[1:].diagonal()
    diag22 = flipboard[:n].diagonal()
    p = 0
    q = 0
    while p < n:
        if diag1[p] == player or diag11[p] == player:
            p += 1
            if p == n:
                value += 1
        else:
            break
    while q < n:
        if diag2[q] == player or diag22[q] == player:
            q += 1
            if q == n:
                value += 1
        else:
            break
    return value


def win_test(player, board, n):
    top_board = board[0:(n*n)]
    matrix = np.reshape(top_board, (n,n))
    diag1 = matrix.diagonal().tolist()
    diag2 = np.fliplr(matrix).diagonal().tolist()
    wins = [diag1, diag2]
    for row in matrix:
        wins.append(row.tolist())
    for i in range(0,n):
        column = []
        for j in range(0,n):
            column.append(top_board[i+j*n])
            wins.append(column)
    for win in wins:
        if len(set(win)) == 1 and win[0] == player:
            return True


def max_value(player, board, alpha, beta, n, path):
    if win_test(player, board, n):
        if len(path) > 1:
            global recommend
            recommend = path[1]
        return value(player, board, n)
    if len(path) >n:
        for move in path:
            if move in ['rotate1', 'rotate2', 'rotate3'] and path.count(move) >= n:
                if len(path) > 1:
                    recommend = path[1]
                return value(player, board, n)
    a = -float('inf')
    successors1 = successors(player, board, n)
    random.shuffle(successors1)
    for successor in successors1:
        path1 = list(path)
        path1.append(successor[0]+str(successor[1]))
        a = max(a, min_value(player, successor[2], alpha, beta, n, path1))
        if a >= beta:
            if len(path) > 1:
                recommend = path[1]
                answer(player, startBoard, n)
            return a
        alpha = max(alpha, a)
    if len(path) > 1:
        recommend = path[1]
    return a

def min_value(player, board, alpha, beta, n, path):
    global recommend
    if win_test(player, board, n):
        if len(path) > 1:
            recommend = path[1]
        return value(player, board, n)
    if len(path) > n:
        for move in path:
            if move in ['rotate1', 'rotate2', 'rotate3'] and path.count(move)>= n:
                if len(path) > 1:
                    recommend = path[1]
                return value(player, board, n)
    a = float('inf')
    successors1 = successors(player, board, n)
    random.shuffle(successors1)
    for successor in successors1:
        path1 = list(path)
        path1.append(successor[0] + str(successor[1]))
        a = min(a, max_value(player, successor[2], alpha, beta, n, path1))
        if a <= alpha:
            if len(path) > 1:
                recommend = path[1]
            return a
        beta = min(beta, a)
    if len(path) > 1:
        recommend = path[1]
    return a

def betsy_solver(player, board, n):
    global recommend
    c = max_value(player, board, alpha, beta, n, ['start'])
    answer(player, board, n)


def pretty_board(board, n):
    for i in range(0, len(board)-1, n):
        print(board[i:i+n])

def answer(player, board, n):
    if recommend[:-1] == 'drop':
        ansBoard = "".join(drop(player, int(recommend[-1:]), board, n))
        print("I'd recommend dropping a pebble in column ",recommend[-1:],". ",ansBoard)
    else:
        ansBoard = "".join(rotate(int(recommend[-1:]), board, n))
        print("I'd recommend rotating column ",recommend[-1:],". ",ansBoard)


betsy_solver(startPlayer, startBoard, n)




