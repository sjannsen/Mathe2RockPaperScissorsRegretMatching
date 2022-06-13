from enum import Enum
import random
import numpy as np
from numpy import ndarray

import matplotlib.pyplot as plt


class Action(Enum):
    ROCK = 0
    PAPER = 1
    SCISSORS = 2


def get_payoff(action_1: Action, action_2: Action) -> int:                          # returns the payoff for player 1
    mod3_val = (action_1.value - action_2.value) % 3
    if mod3_val == 2:
        return -1
    else:
        return mod3_val


def get_strategy(cumulative_regrets: np.array) -> np.array:  # returns the strategy based on regret matching
    pos_cumulative_regrets = np.maximum(0, cumulative_regrets)  # erstellung eines neuen Arrays mit nur positiven regret werten oder 0
    if sum(pos_cumulative_regrets) > 0:
        return pos_cumulative_regrets / sum(pos_cumulative_regrets)                          # return eines Arays mit dem positiven Regret Wert oder 0, geteilt durch 3
    else:
        return np.full(shape=len(Action), fill_value=1 / len(Action))           # wenn alle werte 0 sind, füllen des Arrays mit 1 und teilen durch die 3 Aktionen, also jeder Wert beträgt 1/3


def get_regrets(payoff: int, action_2: Action) -> ndarray:                     # returns the regret
    return np.array([get_payoff(a, action_2) - payoff for a in Action])                                                 # berechnung des payoff für alle möglichen Aktionen gegen die Aktion des Gegners und subtrahieren den eingetretenen payoff
                                                                             # Finally, to compute the regrets, we just compute the payoff for all possible actions against our opponent’s action and subtract the computed payoff:


cumulative_regrets = np.zeros(shape=(len(Action)), dtype=int)                   # [0, 0, 0]
strategy_sum = np.zeros(shape=(len(Action)))                                    # [0, 0, 0]
opponents_strategy = [0.5, 0.2, 0.3]

num_iterations = 10000

probabilityPaper = []
probAllPlot = float

probRockPlot = []
probPaperPlot = []
probScissorsPlot = []

probRock = float
probPaper = float
probScissors = float

for _ in range(num_iterations):
    #  compute the strategy according to regret matching
    strategy = get_strategy(cumulative_regrets)

    #  add the strategy to our running total of strategy probabilities
    strategy_sum += strategy

    # Choose our action and our opponent's action
    our_action = random.choices(list(Action), weights=strategy)[0]
    # opponents_action = random.choices(list(Action), weights=opponents_strategy)[0]  # Against a choosen strategy
    opponents_action = random.choices(list(Action), weights=strategy)[0]  # Against the algorithm itself

    #  compute the payoff and regrets
    our_payoff = get_payoff(our_action, opponents_action)
    regrets = get_regrets(our_payoff, opponents_action)

    #  add regrets from this round to the cumulative regrets
    cumulative_regrets += regrets

    # for creating plots
    probAllPlot = strategy_sum / (_ + 1)
    probabilityPaper.append(probAllPlot)

    probRock = strategy_sum[0] / (_ + 1)
    probPaper = strategy_sum[1] / (_ + 1)
    probScissors = strategy_sum[2] / (_ + 1)

    probRockPlot.append(probRock)
    probPaperPlot.append(probPaper)
    probScissorsPlot.append(probScissors)

optimal_strategy = strategy_sum / num_iterations

np.set_printoptions(formatter={'float_kind': '{:f}'.format})  # to avoid scientific notations which mess up the format

print("\nOptimal_strategy:")
print(optimal_strategy)

# plotting the points
# plt.plot(probabilityPaper, color = "mangenta", label = "")
#
plt.plot(probRockPlot, color="red", label="Rock")
plt.plot(probPaperPlot, color="blue", label="Paper")
plt.plot(probScissorsPlot, color="green", label="Scissors")

plt.xlabel('Number of iterations')
plt.ylabel('Probability')

plt.title('Regret Matching Strategy')

plt.xticks(np.arange(0, len(probabilityPaper) + 1, 1000))
plt.yticks(np.arange(0, 1, 0.2))

plt.legend(loc="upper right")

plt.show()

