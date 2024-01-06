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
    global_cells = cur_state.global_cells.copy()
    for i in range(9):
        if cur_state.game_result(cur_state.blocks[i])==0:
            cur_state.global_cells[i] = 3
    global_cells = convert_to_uttt_index(global_cells)
    blocks = [x for block in cur_state.blocks for b in block for x in b]
    blocks = convert_to_uttt_index(blocks)
    player_to_move = 1 if cur_state.player_to_move==1 else 2
    constraint_index = cur_state.previous_move.x*3 + cur_state.previous_move.y if cur_state.previous_move is not None else 9
    global_constraint_index = constraint_index*9
    if constraint_index < 9 and all(b!=0 for b in blocks[global_constraint_index: global_constraint_index + 9]):
        constraint_index = 9
    if cur_state.game_result(cur_state.global_cells.reshape(3, 3)) == 1:
        result = 1
    elif cur_state.game_result(cur_state.global_cells.reshape(3, 3)) == -1:
        result = 2
    elif cur_state.game_result(cur_state.global_cells.reshape(3, 3)) == 0:
        result = 3
    else:
        result = None
    # result = cur_state.game_result(cur_state.global_cells.reshape(3, 3))
    uttt_state = blocks.copy()
    uttt_state.extend(global_cells)
    uttt_state.extend([player_to_move])
    uttt_state.extend([constraint_index])
    uttt_state.extend([result])
    uttt = UltimateTicTacToe(uttt_state)
    return uttt

def select_move(cur_state, remain_time):
    #Convert to uttt format
    uttt = convert_to_uttt_format(cur_state)
    if uttt.state[91]==9:
        cur_state.free_move = True
    else:
        cur_state.free_move = False
    mcts = MonteCarloTreeSearch(uttt, num_simulations=10000, exploration_strength=1.0)
    mcts.run()
    #Evaluate action
    evaluated_actions = mcts.get_evaluated_actions()
    selected_action = mcts.select_action(evaluated_actions, selection_method="argmax")
    uttt.execute(action=selected_action)
    mcts.synchronize(uttt)
    index_local_board = selected_action.index//9
    x, y = divmod(selected_action.index - index_local_board*9, 3)
    #Revert back to UltimateTTT_Move
    move = UltimateTTT_Move(
        index_local_board=index_local_board,
        x_coordinate=x,  
        y_coordinate=y,
        value= cur_state.player_to_move
    )
    

    return move


