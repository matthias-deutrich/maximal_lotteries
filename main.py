import numpy as np
import cvxpy as cp
from random import randrange


def create_random_wmg(alternative_count,
                      max_margin=100,
                      ensure_oddity=False,
                      exclude_weak_condorcet_winner=False,
                      exclude_weak_condorcet_loser=False):
    assert alternative_count >= 1 and max_margin >= 1
    if ensure_oddity and not max_margin % 2:
        max_margin -= 1
    weighted_majority_graph = np.zeros((alternative_count, alternative_count), dtype=int)
    for i in range(1, alternative_count):
        for j in range(i):
            margin = randrange(-max_margin, max_margin + 1, 2)\
                if ensure_oddity else randrange(-max_margin, max_margin + 1)
            weighted_majority_graph[i, j] = margin
            weighted_majority_graph[j, i] = -margin

    # If the weighted majority graph contains an unwanted condorcet winner/loser, try again
    if exclude_weak_condorcet_winner:
        for i in range(alternative_count):
            if np.logical_and.reduce(weighted_majority_graph[i, :] >= 0):
                return create_random_wmg(alternative_count,
                                         max_margin,
                                         ensure_oddity,
                                         exclude_weak_condorcet_winner,
                                         exclude_weak_condorcet_loser)
    if exclude_weak_condorcet_loser:
        for i in range(alternative_count):
            if np.logical_and.reduce(weighted_majority_graph[i, :] <= 0):
                return create_random_wmg(alternative_count,
                                         max_margin,
                                         ensure_oddity,
                                         exclude_weak_condorcet_winner,
                                         exclude_weak_condorcet_loser)
    return weighted_majority_graph


def maximin(game_matrix, print_output=False):
    assert game_matrix.ndim == 2
    shape = game_matrix.shape
    security_level = cp.Variable()
    strategy = cp.Variable(shape[0])

    objective = cp.Maximize(security_level)
    constraints = [sum(cp.multiply(strategy, game_matrix[:, j])) >= security_level for j in range(shape[1])]
    constraints.append(strategy >= 0)
    constraints.append(sum(strategy) == 1)

    prob = cp.Problem(objective, constraints)
    prob.solve()
    security_level = round(prob.value, 3)
    strategy = [round(p, 3) for p in strategy.value]
    if print_output:
        print('Status: ', prob.status)
        print('Security level: ', security_level)
        print('Maximin strategy: ', strategy)
    return prob.status, security_level, strategy


def shapley_saddle():
    pass


# Compute a maximal lottery. When multiple exist, an arbitrary one may be returned
def compute_maximal_lottery(margin_matrix, tau=lambda x: x):
    # TODO assure that tau is sound, that is that tau(1) = 1 and tau is odd and monotone
    margin_matrix = np.vectorize(tau, otypes=[int])(margin_matrix)
    status, security_level, lottery = maximin(margin_matrix)
    if status != 'optimal':
        raise Exception(f'The status of the problem was {status} instead of \'optimal\'.')
    if security_level != 0.:
        raise Exception(f'The security level was {security_level} instead of 0, so something went wrong.')
    return lottery


def maximal_lottery_power(margin_matrix, power=1, print_output=False):
    if not isinstance(power, int) or power < 1 or power % 2 != 1:
        raise Exception(f'The power {power} does not result in a sound function tau.')
    lottery = compute_maximal_lottery(margin_matrix, tau=lambda x: x ** power)
    if print_output:
        superscript = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
        print(f'For \u03C4(x) = x{"" if power == 1 else str(power).translate(superscript)}, '
              f'a maximal lottery is {lottery}')
    return lottery


sample_game = np.array([
    [0, 1, -3, -3],
    [-1, 0, 1, 11],
    [3, -1, 0, -3],
    [3, -11, 3, 0]
])

interesting_game_1 = np.array([
    [0, -67, 89, 17],
    [67, 0, -91, 12],
    [-89, 91, 0, -21],
    [-17, -12, 21, 0]
])

# Crashed with power 7
# [[  0 -31  97   3]
#  [ 31   0 -66  45]
#  [-97  66   0 -90]
#  [ -3 -45  90   0]]

# Interesting
# [[  0 -51  89 -79]
#  [ 51   0 -75  82]
#  [-89  75   0 -15]
#  [ 79 -82  15   0]]


if __name__ == '__main__':
    game = create_random_wmg(4, exclude_weak_condorcet_winner=True)
    # game = sample_game
    # game = interesting_game_1
    print(game)
    maximal_lottery_power(game, 1, True)
    maximal_lottery_power(game, 3, True)
    maximal_lottery_power(game, 5, True)
