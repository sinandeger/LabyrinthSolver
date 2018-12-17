import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import keras

from tqdm import tqdm

import random
import maze_learner as maze_network

player_marker = 'P'

"""Feed [1, 0, 0] for W, [0, 1, 0] for A, and [0, 0, 1] for D to the neural network"""

"""The current version of the game is designed for NxN boards only.
Future implementation might upgrade the code to run for sides of unequal lengths"""


def move(current_x_pos, current_y_pos, labyrinth, p_marker, board_dim):

    print 'Allowed moves: W (forward), A (left), or D (right)'
    player_move = str(raw_input(">> "))

    lab_borders = labyrinth.shape
    feed_move = []

    print player_move

    if player_move not in ('W', 'w', 'A', 'a', 'D', 'd'):

        print 'Incorrect move'
        move(current_x_pos, current_y_pos, labyrinth, p_marker, board_dim)

    elif player_move == 'W':

        print 'Forward'
        check_within_bounds(current_x_pos, current_y_pos, labyrinth, current_x_pos, current_y_pos-1, board_dim, p_marker)
        print 'Forward check if fall'
        outcome = check_eligible(labyrinth, current_x_pos, current_y_pos-1)
        labyrinth[current_y_pos-1, current_x_pos] = p_marker
        labyrinth[current_y_pos, current_x_pos] = '+'
        #player_move = None
        feed_move = [1, 0, 0, outcome]
        print labyrinth
        print player_move

    elif player_move == 'A':

        print 'Left'
        check_within_bounds(current_x_pos, current_y_pos, labyrinth, current_x_pos-1, current_y_pos, board_dim, p_marker)
        outcome = check_eligible(labyrinth, current_x_pos-1, current_y_pos)
        labyrinth[current_y_pos, current_x_pos-1] = p_marker
        labyrinth[current_y_pos, current_x_pos] = '+'
        feed_move = [0, 1, 0, outcome]
        print labyrinth

    elif player_move == 'D':

        print 'Right'
        check_within_bounds(current_x_pos, current_y_pos, labyrinth, current_x_pos+1, current_y_pos, board_dim, p_marker)
        print 'Right check if fall', player_move
        outcome = check_eligible(labyrinth, current_x_pos+1, current_y_pos)
        print 'passed'
        labyrinth[current_y_pos, current_x_pos+1] = p_marker
        labyrinth[current_y_pos, current_x_pos] = '+'
        feed_move = [0, 0, 1, outcome]
        print labyrinth, feed_move

    print player_move

    print 'Move complete'
    return labyrinth, feed_move


def check_within_bounds(current_x_pos, current_y_pos, labyrinth, x_pos_to_be, y_pos_to_be, board_dim, p_marker):

    print 'Checking whether move within bounds'

    if x_pos_to_be == board_dim or y_pos_to_be == board_dim:

        print 'Move lands you outside the bounds of the board, try again.'
        print 'Current (x,y), board, (x,y)-to-be', current_x_pos, current_y_pos, board_dim, x_pos_to_be, y_pos_to_be
        # player_move = None
        move(current_x_pos, current_y_pos, labyrinth, p_marker, board_dim)

    else:
        print 'Within bounds', current_x_pos, current_y_pos, x_pos_to_be, y_pos_to_be
        pass


def check_eligible(labyrinth, x_pos_to_be, y_pos_to_be):

    print 'Checking whether you fell or not.'

    print 'Your position-to-be', x_pos_to_be, y_pos_to_be

    if labyrinth[y_pos_to_be, x_pos_to_be] == 'x':

        print 'You fell!'
        outcome_feed = -1
        quit()

    else:
        print "You didn't fall"

        outcome_feed = 1
        pass

    return outcome_feed


# def check_eligible(current_x_pos, current_y_pos, labyrinth, x_pos_to_be, y_pos_to_be, board_dim, p_marker):
#
#     print current_x_pos, current_y_pos, x_pos_to_be, y_pos_to_be
#
#     if x_pos_to_be == board_dim or y_pos_to_be == board_dim:
#
#         print 'Move lands you outside the bounds of the board, try again.'
#         print current_x_pos, current_y_pos, board_dim
#         move(current_x_pos, current_y_pos, labyrinth, p_marker, board_dim)
#
#     if labyrinth[y_pos_to_be, x_pos_to_be] == 'x':
#
#         print 'You fell!'
#         quit()
#
#     else:
#         pass


def labyrinth_structure():

    """Need a random labyrinth generator function later"""

    board_dimension = 5
    field = np.empty((board_dimension, board_dimension), dtype=str)

    for k in range(board_dimension):
        for j in range(board_dimension):

            field[k, j] = '+'

    field[4, :3] = 'x'
    field[3, 1:3] = 'x'
    field[0, 0] = 'o'
    field[1, 2:] = 'x'
    field[0, 1:3] = 'x'

    return field


def three_tile_feed(labyrinth, current_x_pos, current_y_pos):

    #next_tiles = np.empty(3, dtype=int)
    next_tiles = []

    """feed[0] for forward, feed[1] for left, feed[2] for right. 1 for a '+'-tile, 0 for either 'x' or out-of-bounds"""

    """Check the forward tile"""
    if current_y_pos-1 < labyrinth.shape[0]:

        if labyrinth[current_y_pos-1, current_x_pos] == '+':
            next_tiles.append(1)

        else:
            next_tiles.append(0)

    else:
        next_tiles.append(0)

    """Check the left tile"""
    if current_x_pos-1 < labyrinth.shape[0]:

        if labyrinth[current_y_pos, current_x_pos-1] == '+':
            next_tiles.append(1)

        else:
            next_tiles.append(0)

    else:
        next_tiles.append(0)

    """Check the right tile"""
    if current_x_pos+1 < labyrinth.shape[0]:

        if labyrinth[current_y_pos, current_x_pos+1] == '+':
            next_tiles.append(1)

        else:
            next_tiles.append(0)

    else:
        next_tiles.append(0)

    tile_forward = (current_y_pos-1, current_x_pos)
    tile_left = (current_y_pos, current_x_pos-1)
    tile_right = (current_y_pos, current_x_pos+1)

    return next_tiles


def check_win(current_x_pos, current_y_pos, win_pos):

    if (current_x_pos, current_y_pos) != (win_pos, win_pos):
        print 'You no won'
        player_wins = 0
        pos_val = 0.5

    else:

        print 'You won!'
        player_wins = 1
        pos_val = 1

    return player_wins, pos_val

#print lab

#start_pos =


def random_walker():

    print 'Not all those who wander are lost.'

#
# def gamestart(field, player_marker):
#
#     field[0, 0] == player_marker


def game_start():

    pass #Later on you need to define a function here to let the player choose the board dimensions.

lab = labyrinth_structure()
print lab.shape

board_dim = 5
lab[board_dim-1, board_dim-1] = player_marker
print lab

win_coord = 0
y_coord_init, x_coord_init = np.where(lab == player_marker)

x_coord, y_coord = int(x_coord_init), int(y_coord_init)

print x_coord, y_coord
player_won, position_value = check_win(x_coord, y_coord, win_coord)

example_games = open('example_games.txt', 'a')

while player_won == 0:

    print 'Turn start'

    next_turn = three_tile_feed(lab, x_coord, y_coord)
    print 'The three possible tiles next turn', next_turn

    lab, outcome_list = move(x_coord, y_coord, lab, player_marker, board_dim)

    print outcome_list

    feed_neural_net = next_turn + outcome_list
    print 'This will be provided to the neural net as a training example turn', feed_neural_net

    print 'Move fine'

    y_coord_init, x_coord_init = np.where(lab == player_marker)
    x_coord, y_coord = int(x_coord_init), int(y_coord_init)

    print x_coord, y_coord
    player_won, position_value = check_win(x_coord, y_coord, win_coord)

    maze_to_feed = np.reshape(lab, (lab.shape[0]*lab.shape[1], 1))
    maze_to_write = np.reshape(lab, (1, lab.shape[0]*lab.shape[1]))

    print >>example_games, maze_to_write, position_value

    #print maze_to_feed

    print 'Turn end'

    # y_coord, x_coord = np.where(lab == player_marker)
    # player_won = check_win(x_coord, y_coord, win_coord)
