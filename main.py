from functools import reduce
from itertools import permutations

import numpy as np
import cvxpy as cp
from random import randrange, shuffle, choice

import sympy
from sympy import Matrix, pprint, init_printing

from sample_games import InterestingGames
from game_matrices import GameMatrix
from utils import *


color_dict = {
    0: '\033[94m',
    1: '\033[96m',
    2: '\033[92m',
    3: '\033[93m',
    -1: '\033[0m'
}


class CustomException(Exception):
    pass


def matrix_from_margins_list(margins_list):
    weighted_majority_graph = np.zeros((len(margins_list + 1), len(margins_list + 1)), dtype=int)
    for i, margins in enumerate(margins_list):
        for j, margin in enumerate(margins, start=i + 1):
            weighted_majority_graph[i][j] = margin
            weighted_majority_graph[j][i] = -margin
    return weighted_majority_graph


def create_random_wmg(alternative_count,
                      max_margin=30,
                      ensure_uniqueness=False,
                      ensure_oddity=False,
                      exclude_weak_condorcet_winner=False,
                      exclude_weak_condorcet_loser=False,
                      exclude_zero=False):
    assert alternative_count >= 1 and max_margin >= 1

    def non_unique_rng():
        while True:
            yield randrange(-max_margin + (1 if ensure_oddity and max_margin % 2 == 0 else 0),
                             max_margin + 1,
                             2 if ensure_oddity else 1)

    def unique_rng():
        numbers = list(range(1 if ensure_oddity else 0, max_margin + 1, 2 if ensure_oddity else 1))
        shuffle(numbers)
        while numbers:
            yield choice([-1, 1]) * numbers.pop()

    rng = unique_rng() if ensure_uniqueness else non_unique_rng()

    weighted_majority_graph = np.zeros((alternative_count, alternative_count), dtype=int)
    for i in range(1, alternative_count):
        for j in range(i):
            margin = next(rng)
            while margin == 0 and exclude_zero:
                margin = next(rng)
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


# This takes as an input a list of tuples, which represents the order of edge weights, from lowest to highest.
# An edge between i and j is represented by the tuple (i, j). Uniqueness and completeness must be ensured by the user.
# Returns a weighted majority graph where the weights are in the given order.
def create_graph_specific_ordered_wmg(order, max_margin=30, randomize_order=False, print_game=False):
    # Create the raw values
    vals = list(range(1, max_margin + 1, 2))
    shuffle(vals)
    vals = vals[:len(order)]
    if not randomize_order:
        vals.sort()
    if print_game:
        print(f'The created game has the values {vals}')

    # Create and fill the matrix
    alternative_count = max([max(pair) for pair in order]) + 1
    weighted_majority_graph = np.zeros((alternative_count, alternative_count), dtype=int)
    for i, j in order:
        val = vals.pop(0)
        weighted_majority_graph[i, j] = val
        weighted_majority_graph[j, i] = -val
    return weighted_majority_graph


# Takes as an input any function on the reals and returns an odd function wich is identical for positive numbers
# TODO give option to normalize (assuring tau(0) = 0 and tau(1) = 1
def oddify(tau):
    return lambda x: tau(x) if x >= 0 else -tau(-x)


def maximin_cvxpy(game_matrix, print_output=False):
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


def t_game(margin_matrix, t):
    return np.vectorize(lambda x: x ** t if x > 0 else - (-x) ** t)(margin_matrix)


# Compute a maximal lottery. When multiple exist, an arbitrary one may be returned
def compute_maximal_lottery(margin_matrix, tau=lambda x: x, oddify_tau=True):
    # TODO assure that tau is sound, that is that tau(1) = 1 and tau is odd and monotone
    margin_matrix = np.vectorize(oddify(tau) if oddify_tau else tau, otypes=[int])(margin_matrix)
    result = maximin(margin_matrix)
    if not result[0]:
        raise CustomException(f'The problem could not be solved with the following reason: {result[1]}')
    else:
        security_level = round(result[1], 3)
        slack = [round(prob, 3) for prob in result[3]]
        lottery = [round(prob, 3) for prob in result[2]]
        if security_level != 0.:
            raise CustomException(f'The security level was {security_level} instead of 0, so something went wrong.')
        return lottery, slack


def maximal_lottery_power(margin_matrix, power=1, print_output=False, print_slack=False):
    if not isinstance(power, int) or power < 1:
        raise CustomException(f'The power {power} does not result in a sound function tau.')
    lottery, slack = compute_maximal_lottery(margin_matrix, tau=lambda x: x ** power, oddify_tau=True)
    if print_output:
        superscript = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
        print(f'For \u03C4(x) = x{"" if power == 1 else str(power).translate(superscript)}, '
              f'a maximal lottery is {lottery}' + (f' with slack {slack}' if print_slack else ''))
    return lottery, slack


# TODO potentially include 0, see if this causes problems with monotonicity
def power_sequence(
        margin_matrix,
        max_power=10,
        print_output=False,
        print_slack=False
):
    lotteries, slacks = [], []
    for t in range(1, max_power + 1):
        try:
            lottery, slack = (maximal_lottery_power(
                margin_matrix=margin_matrix,
                power=t,
                print_output=print_output,
                print_slack=print_slack
            ))
            lotteries.append(lottery)
            slacks.append(slack)
        except CustomException as e:
            if print_output:
                print(e)
            break

    lotteries = np.asarray(lotteries)
    slacks = np.asarray(slacks)
    return lotteries, slacks


def check_for_monotonicity(
        lotteries,
        semi_monotonicity=False,
        print_output=False
):
    monotone = True
    for j in range(lotteries.shape[1]):
        if semi_monotonicity:
            curr_mono = True
            falling = False
            last_val = lotteries[0, j]
            for val in lotteries[1:, j]:
                if val < last_val:
                    falling = True
                if val > last_val and falling:
                    curr_mono = False
                last_val = val
        else:
            curr_mono = np.all(np.equal(lotteries[:, j], sorted(lotteries[:, j]))) \
                        or np.all(np.equal(lotteries[:, j], sorted(lotteries[:, j], reverse=True)))
        monotone &= curr_mono
        if print_output and not monotone:
            raise CustomException(f'The {j}th sequence of probabilities is {lotteries[:, j]}, which is not sorted.')
    return monotone


def check_mass_mono(
        game_count,
        alt_count=4,
        max_power=10,
        semi_monotonicity=False,
        print_output=False
):
    mono_violation_count = 0
    for _ in range(game_count):
        curr_game = create_random_wmg(alt_count, max_margin=30, ensure_oddity=True, exclude_weak_condorcet_winner=True)
        lotteries, _ = power_sequence(
            margin_matrix=curr_game,
            max_power=max_power,
            print_output=False
        )
        mono_violation_count += 0 if check_for_monotonicity(lotteries, semi_monotonicity=semi_monotonicity) else 1
    if print_output:
        print(f'Checked {game_count} games, {mono_violation_count} of which were not monotonous.')
    return mono_violation_count


def winner(lotteries, threshold=0.55, print_output=False):
    winning_alternative = np.argmax(lotteries, 1)[-1]
    winning_prob = np.max(lotteries, 1)[-1]
    if threshold:
        if winning_prob < threshold:
            raise CustomException(f'The winning alternative in the last lottery was only {winning_prob}')
    if print_output:
        print(f'The last lottery was won by alternative {winning_alternative} '
              'with sufficient probability to indicate convergence.')
    return winning_alternative


def all_slacking_alts(slacks):
    positive_slacks = np.greater_equal(slacks, 0.001)
    positive_vars = np.any(positive_slacks, axis=0)
    return set(np.where(positive_vars)[0])


def last_slacking_alt(slacks):
    positive_slacks = np.greater_equal(slacks[:][-1], 0.001)
    positive_indices = np.where(positive_slacks)[0]
    if len(positive_indices) > 1:
        raise CustomException(f'All equations {positive_indices} had slack, which is more than 1')
    return positive_indices[0]


def slack_winner_combination(max_margin=30, print_output=False, print_game=False):
    positive_margins = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (3, 0)]
    weighted_majority_graph = create_graph_specific_ordered_wmg(positive_margins,
                                                                max_margin=max_margin,
                                                                randomize_order=True,
                                                                print_game=print_game)
    lotteries, slacks = power_sequence(weighted_majority_graph)
    try:
        winning_alternative = winner(lotteries)
    except CustomException as e:
        print(weighted_majority_graph)
        print(lotteries[:][-1])
        print(e, '\n')
        winning_alternative = None
    slackers = last_slacking_alt(slacks)
    if print_output:
        print(f'{color_dict[slackers]}The lottery tends to {winning_alternative}',
              f'and final slackers were {slackers}{color_dict[-1]}')
    return winning_alternative, slackers


def mass_winner_slacker_combination_check(game_count=100, alternative_count=4, max_margin=30,
                                          print_output=False, print_game=False):
    combination_set = set()
    for _ in range(game_count):
        combination_set.add(slack_winner_combination(max_margin=max_margin,
                                                     print_output=print_output,
                                                     print_game=print_game))
    if print_output:
        print(f'All occurring combinations of (winner, slack) were {combination_set}')


def compute_winners_of_order(order, game_count=100, max_margin=30, threshold=None,
                             print_simplified_output=False, print_full_output=False,
                             print_slack=False, print_game=False):
    winners = set()
    for _ in range(game_count):
        weighted_majority_graph = create_graph_specific_ordered_wmg(order, max_margin=max_margin, print_game=print_game)
        lotteries, slacks = power_sequence(weighted_majority_graph)
        winning_alternative = winner(lotteries, threshold=threshold)
        winners.add(winning_alternative)
        if print_simplified_output:
            print(f'{color_dict[winning_alternative]}The lottery tends to {winning_alternative},',
                  f'all alternatives that slacked in computation are {all_slacking_alts(slacks)}\033[0m')
        if print_full_output:
            print(f'{color_dict[winning_alternative]}The final lottery of the current game is {lotteries[-1][:]}',
                  (f'\tSlack is {slacks[-1][:]}' if print_slack else ''),
                  '\033[0m')
    return winners


def check_whether_winner_is_unique_to_order(max_margin=30, fail_on_multiple=False, print_output=False):
    positive_margins = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (3, 0)]
    for permutation in permutations(positive_margins):
        winners = compute_winners_of_order(permutation, max_margin=max_margin)
        if print_output:
            print(('\033[91m' if len(winners) > 1 else '\033[94m') + f'The winners of permutation {permutation} are {winners}.\033[0m')
        if fail_on_multiple and len(winners) > 1:
            raise CustomException(f'Permutation {permutation} produced multiple winners {winners}!')


def linalg_from_game(matrix, indices: list = None):
    matrix = np.asarray(matrix)
    if indices:
        matrix = matrix[np.ix_(indices, indices)]
    matrix = np.append(matrix, np.zeros((matrix.shape[0], 1), dtype=int), axis=1)
    return np.append(matrix, np.ones((1, matrix.shape[1]), dtype=int), axis=0)


def rref_from_game(matrix, indices=None):
    matrix = Matrix(linalg_from_game(matrix, indices))
    return matrix.rref()


def rref_valid(matrix, indices=None, print_output=False):
    alt_count = len(matrix[0])
    rref = rref_from_game(matrix, indices)
    if print_output:
        print(rref[0])
    vals = rref[0][alt_count:(alt_count + 1)**2 - 1:alt_count + 1]
    is_valid = reduce((lambda x, y: x and y), [val >= 0 for val in vals])
    return is_valid and rref[0][-1] == 0


def always_rank_minus_1_test(n, count=1000):
    for i in range(count):
        matrix = create_random_wmg(n, max_margin=30)
        if np.linalg.matrix_rank(matrix) != n - 1:
            raise Exception(matrix)


normalized_four_game_edges = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (3, 0)]


if __name__ == '__main__':

    # game = create_random_wmg(4, max_margin=30, ensure_oddity=True, exclude_weak_condorcet_winner=True)
    # tail = [(3, 0), (1, 2), (2, 3), (0, 1), (1, 3)]
    # shuffle(tail)
    # game = create_graph_specific_four_wmg([(0, 2)] + tail)
    # game = sample_game
    # game = interesting_game_1
    # game = semi_mono_violation
    # game = create_graph_specific_ordered_wmg([(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (3, 0)], max_margin=11)
    # game = InterestingGames.simple_test_game

    # game = InterestingGames.same_order_different_winner_0
    # print(game)
    # power_sequence(game, print_output=True, print_slack=True)
    # print('\n\n')
    # game = InterestingGames.same_order_different_winner_1
    # print(game)
    # power_sequence(game, print_output=True, print_slack=True)

    # game = InterestingGames.non_mono_at_3
    # print(game)
    # power_sequence(game, print_output=True, print_slack=True)

    # game = create_random_wmg(
    #     alternative_count=5,
    #     max_margin=30,
    #     ensure_uniqueness=False,
    #     ensure_oddity=True,
    #     exclude_weak_condorcet_winner=True,
    #     exclude_weak_condorcet_loser=True
    # )
    # always_rank_minus_1_test(5, 100000)

    # t = sympy.Symbol('t')
    # m = Matrix([[1, 2, t], [0, 1, 2**t], [0, 0, 0]])
    # print(m.rref())

    pprint(InterestingGames.chris_example.rref())


    game = np.asarray(InterestingGames.chris_example)

    # print(game)
    # power_sequence(game, max_power=10, print_output=True, print_slack=True)
    # print(rref_from_game(game))
    # print(rref_from_game(t_game(game, 2)))
    # print(rref_from_game(t_game(game, 3)))

    # print(Matrix(t_game(game, 4)).rref())

    # game = InterestingGames.five_alts_no_slackers
    # game = t_game(game, 1)
    # print(rref_valid(game, None, True))
    # game = linalg_from_game(game, [0, 2])
    # game = create_graph_specific_ordered_wmg([(2, 3), (0, 2), (0, 1), (3, 0), (1, 2), (1, 3)], max_margin=11)
    # game = InterestingGames.five_alts_no_slackers
    # print(game)
    # power_sequence(game, max_power=10, print_output=True, print_slack=True)

    # print(game)
    # power_sequence(game, print_output=True, print_slack=True)
    # lotteries = power_sequence(
    #     game,
    #     check_for_monotonicity=False,
    #     print_output=True
    # )
    # winner(lotteries, print_output=True)

    # semi_monotonicity_check = True
    # mono = power_sequence(
    #     game,
    #     check_for_monotonicity=True,
    #     semi_monotonicity=semi_monotonicity_check,
    #     raise_error_on_mono_failure=True,
    #     print_output=True
    # )
    # if mono:
    #     print(f'The lotteries are {"semi-" if semi_monotonicity_check else ""}monotonous.')
    # else:
    #     print(f'The lotteries are not {"semi-" if semi_monotonicity_check else ""}monotonous!', file=sys.stderr)
    # check_mass_mono(10000, alt_count=4, semi_monotonicity=True, raise_error_on_mono_failure=False, print_output=False)

    # check_whether_winner_is_unique_to_order(max_margin=100, fail_on_multiple=False, print_output=True)
    # The winners of permutation((0, 1), (1, 2), (0, 2), (1, 3), (3, 0), (2, 3)) are {0, 1}.
    # The winners of permutation((0, 1), (1, 2), (1, 3), (0, 2), (3, 0), (2, 3)) are {0, 1}.

    # compute_winners_of_order(order=[(0, 1), (1, 2), (0, 2), (1, 3), (3, 0), (2, 3)], max_margin=15, print_full_output=True)
    # compute_winners_of_order(order=[(1, 2), (2, 3), (1, 3), (0, 2), (3, 0), (0, 1)], max_margin=13,
    #                         print_simplified_output=True, print_full_output=True, print_slack=True, print_game=True)

    # mass_winner_slacker_combination_check(game_count=1000, print_output=True, print_game=True)
