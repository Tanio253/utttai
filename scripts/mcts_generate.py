import argparse
import pathlib
import random
import sys
sys.path.insert(0, '/home/tanio/nam3/ML/utttai')
from utttpy.selfplay.monte_carlo_tree_search import (
    MonteCarloTreeSearch,
    serialize_evaluated_state,
    serialize_evaluated_actions,
)
from utttpy.game.ultimate_tic_tac_toe import UltimateTicTacToe


def run_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # parser.add_argument("--uttt_state", type=str, required=True)
    parser.add_argument("--num_simulations", type=int, required=True)
    parser.add_argument("--exploration_strength", type=float, required=True)
    parser.add_argument("--random_seed", type=int, required=True)
    parser.add_argument("--output_path", type=pathlib.Path, required=True)
    args = parser.parse_args()
    return args


def main() -> None:
    args = run_argparse()
    print(args)

    random.seed(args.random_seed)

    uttt = UltimateTicTacToe()

    mcts = MonteCarloTreeSearch(
        uttt=uttt.clone(),
        num_simulations=args.num_simulations,
        exploration_strength=args.exploration_strength,
    )
    evaluations_str = ""
    while not uttt.is_terminated():
        mcts.run(progress_bar=True)
        print(mcts)
        evaluated_state = mcts.get_evaluated_state()
        evaluated_actions = mcts.get_evaluated_actions()
        evaluated_state_str = serialize_evaluated_state(evaluated_state=evaluated_state)
        evaluated_actions_str = serialize_evaluated_actions(evaluated_actions=evaluated_actions)
        evaluation_str = f"{evaluated_state_str} {evaluated_actions_str}"
        print(evaluation_str)
        evaluations_str += f"{evaluation_str}\n"
        selected_action = mcts.select_action(evaluated_actions=evaluated_actions, selection_method="sample")
        print("selected", selected_action)
        uttt.execute(action=selected_action)
        mcts.synchronize(uttt=uttt)
    print(mcts)

    with open(args.output_path, "w") as f:
        f.write(evaluations_str)
    print(f"evaluations saved to {args.output_path} successfully!")


if __name__ == "__main__":
    main()
