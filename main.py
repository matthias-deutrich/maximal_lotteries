import numpy as np
import cvxpy as cp
from random import randrange
import sys


def create_random_wmg(alternative_count,
                      max_margin=50,
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


# Takes as an input any function on the reals and returns an odd function wich is identical for positive numbers
# TODO give option to normalize (assuring tau(0) = 0 and tau(1) = 1
def oddify(tau):
    return lambda x: tau(x) if x >= 0 else -tau(-x)


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
def compute_maximal_lottery(margin_matrix, tau=lambda x: x, oddify_tau=True):
    # TODO assure that tau is sound, that is that tau(1) = 1 and tau is odd and monotone
    margin_matrix = np.vectorize(oddify(tau) if oddify_tau else tau, otypes=[int])(margin_matrix)
    status, security_level, lottery = maximin(margin_matrix)
    if status != 'optimal':
        raise Exception(f'The status of the problem was {status} instead of \'optimal\'.')
    if security_level != 0.:
        raise Exception(f'The security level was {security_level} instead of 0, so something went wrong.')
    return lottery


def maximal_lottery_power(margin_matrix, power=1, print_output=False):
    if not isinstance(power, int) or power < 1:
        raise Exception(f'The power {power} does not result in a sound function tau.')
    lottery = compute_maximal_lottery(margin_matrix, tau=lambda x: x ** power, oddify_tau=True)
    if print_output:
        superscript = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
        print(f'For \u03C4(x) = x{"" if power == 1 else str(power).translate(superscript)}, '
              f'a maximal lottery is {lottery}')
    return lottery


# TODO potentially include 0, see if this causes problems with monotonicity
def power_sequence(
        margin_matrix,
        max_power=6,
        check_for_monotonicity=True,
        raise_error_on_mono_failure=False,
        print_output=False
):
    lotteries = []
    for t in range(1, max_power + 1):
        try:
            lotteries.append(maximal_lottery_power(margin_matrix, t, print_output))
        except Exception as e:
            print(e, file=sys.stderr)
            break
    if check_for_monotonicity:
        monotone = True
        lotteries = np.asarray(lotteries)
        for j in range(lotteries.shape[1]):
            monotone &= np.all(np.equal(lotteries[:, j], sorted(lotteries[:, j])))\
                        or np.all(np.equal(lotteries[:, j], sorted(lotteries[:, j], reverse=True)))
            if raise_error_on_mono_failure and not monotone:
                raise Exception(f'The {j}th sequence of probabilities is {lotteries[:, j]}, which is not sorted.\n'
                                f'Underlying majority margin matrix is:\n{margin_matrix}')
        return monotone
    return None


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

# TODO check for solution uniqueness
# Potential monotonicity violation
# [[  0 -32  44  26]
#  [ 32   0 -42 -45]
#  [-44  42   0  41]
#  [-26  45 -41   0]]

# TODO this was odd, so this might be a real counterexample
# Potential monotonicity violation
# [[  0  33   9 -25]
#  [-33   0  -3  23]
#  [ -9   3   0   5]
#  [ 25 -23  -5   0]]

# Another potential mono violation
# [[  0  37 -31  19]
#  [-37   0 -19  49]
#  [ 31  19   0 -39]
#  [-19 -49  39   0]]

# Ditto
# [[  0  41  41 -47]
#  [-41   0   9  -5]
#  [-41  -9   0  25]
#  [ 47   5 -25   0]]

# Ditto, maximum surprisingly at x^3
# [[  0 -41  45  17]
#  [ 41   0 -29  21]
#  [-45  29   0  17]
#  [-17 -21 -17   0]]

# Ditto, for three alternatives
# [[  0 -13  29]
#  [ 13   0 -25]
#  [-29  25   0]]

# Ditto
# [[  0 -15  13]
#  [ 15   0  -9]
#  [-13   9   0]]

if __name__ == '__main__':
    game = create_random_wmg(3, ensure_oddity=True, exclude_weak_condorcet_winner=True)
    # game = sample_game
    # game = interesting_game_1
    print(game)
    mono = power_sequence(game, check_for_monotonicity=True, raise_error_on_mono_failure=True, print_output=True)
    if mono:
        print('The lotteries are monotonous.')
    else:
        print('The lotteries are not monotonous!', file=sys.stderr)
