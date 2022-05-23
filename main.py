from enum import Enum
import random
from typing import List
import numpy as np
from numpy import ndarray


class Action(Enum):
    ROCK = 0
    PAPER = 1
    SCISSORS = 2


def get_payoff(action_1: Action, action_2: Action) -> int: # returns the payoff for player 1
    mod3_val = (action_1.value - action_2.value) % 3
    if mod3_val == 2:
        return -1
    else:
        return mod3_val


def get_strategy(cumulative_regrets: np.array) -> np.array: # returns the strategy based on regret matching
    pos_cumulative_regrets = np.maximum(0, cumulative_regrets)
    if sum(pos_cumulative_regrets) > 0:
        return pos_cumulative_regrets / sum(pos_cumulative_regrets)
    else:
        return np.full(shape=len(Action), fill_value=1 / len(Action))


def get_regrets(payoff: int, action_2: Action) -> ndarray: # returns the regret
    return np.array([get_payoff(a, action_2) - payoff for a in Action])


cumulative_regrets = np.zeros(shape=(len(Action)), dtype=int)
strategy_sum = np.zeros(shape=(len(Action)))
opponents_strategy = [0.5, 0.2, 0.3]

num_iterations = 10000

for _ in range(num_iterations):
    #  compute the strategy according to regret matching
    strategy = get_strategy(cumulative_regrets)

    #  add the strategy to our running total of strategy probabilities
    strategy_sum += strategy

    # Choose our action and our opponent's action
    our_action = random.choices(list(Action), weights=strategy)[0]
    # opponents_action = random.choices(list(Action), weights=opponents_strategy)[0] #Against a choosen strategy
    opponents_action = random.choices(list(Action), weights=strategy)[0]  # Against the algorithm itself

    #  compute the payoff and regrets
    our_payoff = get_payoff(our_action, opponents_action)
    regrets = get_regrets(our_payoff, opponents_action)

    #  add regrets from this round to the cumulative regrets
    cumulative_regrets += regrets

optimal_strategy = strategy_sum / num_iterations

np.set_printoptions(formatter={'float_kind': '{:f}'.format})  # to avoid scientific notations which mess up the format

print("Strategy_sum: ")
print(strategy_sum)

print("\nOptimal_strategy:")
print(optimal_strategy)
