#!/bin/python

import sys
import numpy as np
import math
import random
import time

# Command line arguments

# Number of columns for the Betsy board. Set variable as integer to allow numerical operations
n = int(sys.argv[1])
# Player whose turn it is.
player = sys.argv[2]
# The current state of the board
board = sys.argv[3]
# How long the program is permitted to run. Set to float to provide more
# resolution to the time control
duration = float(sys.argv[4])

# These time values are used to determine when the program has ran past the allotted time
# Records start time of program
start_time = time.time()
# Updates run_time through each time min_value or max_value is called
run_time = 0

# Initializes alpha and beta to the worst possible values for each, negative infinity and positive
# infinity respectively
alpha = -math.inf
beta = math.inf

# Stores the most recent recommendation while the search algorithm runs
recommend = ''


# Functions used to execute game play

# Drops a pebble for a given player, column, and board
def drop(player, column, board, n):
    board1 = board.copy()
    for i in range((n * (n + 3)) - 1 - (n - column), -1, -n):
        if board1[i] == '.':
            board1[i]= player
            return board1

# Rotates the given column for a given board
def rotate(column, board, n):
    board1 = board.copy()
    for i in range((n * (n + 3)) - 1 - (n - column), -1, -n):
        board1[i] = board[i-n]
    for k in range(0+(column-1), (n*(n+3))-n, n):
        if board1[k] != '.' and board1[k+n] == '.':
            board1[k], board1[k+n] = board1[k+n], board1[k]
    return board1


# Functions utilized for the minimax w/ alpha-beta pruning algorithm

# Given whose turn it is and board, returns a list of all possible moves
def successors(player, board, n):
    actions = []
    board1 = board.copy()
    for i in range(1, n + 1):
        a = drop(player, i, board1, n)
        if a is not None:
            actions.append(['drop', i, a])
        actions.append(['rotate', i, rotate(i, board1, n)])
    return actions

# Heuristic function that calculates the estimated value for a given board for a given player.
# Creates a new board with the top n rows and the bottom row, as these pebbles can be scored if a column
# is rotated. For that new board, for every pair of rows where each column has at least one pebble of the player's color,
# the value is increased by one. Similarly, for every pair of diagonals, if each column has at least one pebble of the
# player's color, the value is increased by one. It also adds 1 to value for every pebble of player's color in the
# new board, and subtracts for every pebble that is the other player's.
def value(player, board, n):
    value = 0
    board1 = (board[-n:] + board[0:(n * n)])
    #print(board1)
    board2 = np.reshape(board1, (n + 1, n))

    # Adds to value for every pair of rows that has the players pebble in each column for at least one row
    i = n # row
    j = 0 # item
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
        #print(value)

    # Adds to value for each pebble that is players and subtracts for each that is not. Goes through by column
    for k in range(0,n): # item
        for m in range(n, -1, -1): # row
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

    # Adds to value if each column from the two defined diagonals contains at least one player. As soon as not, breaks
    while p < n:
        if diag1[p] == player or diag11[p] == player:
            p += 1
            if p == n:
                value += 1
        else:
            break
    # Flipped board to get diagonals pointing the other way
    while q < n:
        if diag2[q] == player or diag22[q] == player:
            q += 1
            if q == n:
                value += 1
        else:
            break
    return value

# Determines if the given board is a terminal state given that it is player's turn
def win_test(player, board, n):
    top_board = board[0:(n*n)]
    matrix = np.reshape(top_board, (n,n))
    diag1 = np.reshape(matrix, (n, n)).diagonal().tolist()
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

# Calculates the best move for the given ('max') player. This is done by first checking if board is a terminal
# state (win) for max. If not, it finds the successors of the given board, and calls min_value() on each successor
# which finds the successor with the lowest utility value, as this is what min would choose. It then compares this value
# with the best current option for max, alpha, and if the successor is better, it becomes the new alpha value. If not,
#  then alpha stays the same, and the process is continued down the tree.
def max_value(player, board, alpha, beta, n, path):

    # makes changes to run_time made in function available outside of function
    global run_time
    # Assigns latest run_time value outside of function
    run_time = time.time() - start_time
    # Tests if program has ran longer than defined duration, and if so, ends program
    if run_time > duration:
        sys.exit('Times up')

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
    a = -math.inf
    successors1 = successors(player, board, n)
    random.shuffle(successors1)
    for successor in successors1:
        path1 = path.copy()
        path1.append(successor[0]+str(successor[1]))
        a = max(a, min_value(player, successor[2], alpha, beta, n, path1))
        if a >= beta:
            if len(path) > 1:
                recommend = path[1]
                answer(player, board, n)
            print('max', a)
            return a
        alpha = max(alpha, a)
    if len(path) > 1:
        recommend = path[1]
    print('max',a)
    return a

# Calculates the best move for the other player ('min'). The best move for 'min' is the one that generates the lowest
# heuristic value.  This is done by first checking if board is a terminal state (win) for max. If not, it finds the
# successors of the given board, and calls max_value() on each successor
# which finds the successor with the highest utility value, as this is what max would choose. It then compares this
# value with the best current option for min, beta (which would the option that minimizes utility value), and if the
# successor is better, it becomes the new beta value. If not, then beta stays the same, and the process is continued
# down the tree.
def min_value(player, board, alpha, beta, n, path):
    global recommend

    # makes changes to run_time made in function available outside of function
    global run_time
    # Assigns latest run_time value outside of function
    run_time = time.time() - start_time
    # Tests if program has ran longer than defined duration, and if so, ends program
    if run_time > duration:
        sys.exit('Times up')

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
    a = math.inf
    successors1 = successors(player, board, n)
    random.shuffle(successors1)
    for successor in successors1:
        path1 = path.copy()
        path1.append(successor[0] + str(successor[1]))
        a = min(a, max_value(player, successor[2], alpha, beta, n, path1))
        if a <= alpha:
            if len(path) > 1:
                recommend = path[1]
            print('min', a)
            return a
        beta = min(beta, a)
    if len(path) > 1:
        recommend = path[1]
    print('min',a)
    return a

# Begins the call of max_value to find solution
def betsy_solver(player, board, n):
    global recommend
    c = max_value(player, board, alpha, beta, n, ['start'])
    print(board)
    answer(player, board, n)

# prints an easily read board
def pretty_board(board, n):
    for i in range(0, len(board)-1, n):
        print(board[i:i+n])

# formats the output to print suggested solution
def answer(player, board, n):
    print('answer_test',drop(player,int(recommend[-1:]), board,n))
    if recommend[:-1] == 'drop':
        ansBoard = "".join(drop(player, int(recommend[-1:]), board, n))
        print("I'd recommend dropping a pebble in column ",recommend[-1:],". ",ansBoard)
    else:
        ansBoard = "".join(rotate(int(recommend[-1:]), board, n))
        print("I'd recommend rotating column ",recommend[-1:],". ",ansBoard)

# calls commandline arguments to run program
betsy_solver(player, list(board), n)




