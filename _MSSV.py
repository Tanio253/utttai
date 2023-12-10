import numpy as np
import time
import math
import random
from state import State
from state import UltimateTTT_Move
from utttpy.game.ultimate_tic_tac_toe import UltimateTicTacToe
from utttpy.selfplay.monte_carlo_tree_search import MonteCarloTreeSearch

def convert_to_uttt_index(li):
    res_list = []
    for l in li:
        if l==-1:
            res_list.append(2)
            continue
        res_list.append(l)
    return res_list        

def convert_to_uttt_format(cur_state):
    # Convert the relevant information from cur_state to UltimateTicTacToe format
    global_cells = cur_state.global_cells.copy()
    global_cells = convert_to_uttt_index(global_cells)
    blocks = [x for block in cur_state.blocks for b in block for x in b]
    blocks = convert_to_uttt_index(blocks)
    player_to_move = 1 if cur_state.player_to_move==1 else 2 
    constraint_index = cur_state.previous_move.x*3 + cur_state.previous_move.y if cur_state.previous_move is not None else 9
    global_constraint_index = constraint_index*9
    # print(global_constraint_index)
    # print(blocks[global_constraint_index: global_constraint_index+9])
    if constraint_index < 9 and all(b!=0 for b in blocks[global_constraint_index: global_constraint_index + 9]):
        constraint_index = 9
    result = cur_state.game_result(cur_state.global_cells.reshape(3, 3))
    uttt_state = blocks.copy()
    uttt_state.extend(global_cells)
    uttt_state.extend([player_to_move])
    uttt_state.extend([constraint_index])
    uttt_state.extend([result])
    # Create an instance of UltimateTicTacToe with the converted information
    uttt = UltimateTicTacToe(uttt_state)
    return uttt

def select_move(cur_state, remain_time):
    # Convert cur_state to UltimateTicTacToe format
    uttt = convert_to_uttt_format(cur_state)

    # Create an instance of MonteCarloTreeSearch
    mcts = MonteCarloTreeSearch(uttt, num_simulations=1000, exploration_strength=1.0)

    # Synchronize the tree with the current state

    # Run Monte Carlo Tree Search to explore potential moves
    mcts.run()

    # Get the evaluated actions from the root node
    evaluated_actions = mcts.get_evaluated_actions()

    # Select the move using Monte Carlo Tree Search results
    selected_action = mcts.select_action(evaluated_actions, selection_method="argmax")

    # Convert the selected action to UltimateTTT_Move format
    uttt.execute(action=selected_action)
    mcts.synchronize(uttt)
    index_local_board = selected_action.index//9
    x, y = divmod(selected_action.index - index_local_board*9, 3)
    move = UltimateTTT_Move(
        index_local_board=index_local_board,
        x_coordinate=x,  # Convert 1D index to 2D coordinates
        y_coordinate=y,
        value= cur_state.player_to_move  # Use the current player's symbol
    )

    return move


