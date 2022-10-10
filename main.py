from itertools import permutations

import numpy as np
import cvxpy as cp
from random import randrange, shuffle, choice
import sys
from scipy.optimize import linprog


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
                      exclude_weak_condorcet_loser=False):
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
def create_graph_specific_ordered_wmg(order, max_margin=30, print_game=False):
    # Create the raw values
    vals = list(range(1, max_margin + 1, 2))
    shuffle(vals)
    vals = vals[:len(order)]
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


def maximin(game_matrix: np.ndarray, print_output=False):
    assert game_matrix.ndim == 2
    shape = game_matrix.shape

    result = linprog(
        c=[0 for i in range(shape[0])] + [-1],
        A_ub=np.append(-1 * np.transpose(game_matrix), np.full((shape[1], 1), 1), axis=1),
        b_ub=np.full(shape[1], 0),
        A_eq=np.append(np.full((1, shape[0]), 1), [[0]], axis=1),
        b_eq=np.asarray(1.),
        bounds=[(0, None) for i in range(shape[0])] + [(None, None)],
        method='highs'
    )

    if result.success:
        security_level = result.x[-1]
        strategy = result.x[:-1]

        if print_output:
            print('Security level: ', security_level)
            print('Maximin strategy: ', strategy)
        return True, security_level, strategy, result.slack
    else:
        return False, result.message


def shapley_saddle():
    pass


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


def slack_winner_combination(max_margin=30, print_output=False):
    positive_margins = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (3, 0)]
    shuffle(positive_margins)
    weighted_majority_graph = create_graph_specific_ordered_wmg(positive_margins, max_margin=max_margin, print_game=False)
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
        print(f'The lottery tends to {winning_alternative} and final slackers were {slackers}')
    return winning_alternative, slackers


def mass_winner_slacker_combination_check(game_count=100, alternative_count=4, max_margin=30, print_output=False):
    combination_set = set()
    for _ in range(game_count):
        combination_set.add(slack_winner_combination(max_margin=max_margin, print_output=print_output))
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
        match winning_alternative:
            case 0:
                color = '\033[94m'
            case 1:
                color = '\033[96m'
            case 2:
                color = '\033[92m'
            case _:
                color = '\033[93m'
        if lotteries[-1][winning_alternative] < 0.6:
            color = '\033[91m'
        if print_simplified_output:
            print(f'{color}The lottery tends to {winning_alternative},',
                  f'all alternatives that slacked in computation are {all_slacking_alts(slacks)}\033[0m')
        if print_full_output:
            print(f'{color}The final lottery of the current game is {lotteries[-1][:]}',
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


class InterestingGames:
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
    non_mono_at_3 = [
        [0, -41, 45, 17],
        [41, 0, -29, 21],
        [-45, 29, 0, 17],
        [-17, -21, -17, 0]
    ]

    # Ditto, for three alternatives
    mono_violation_game_1 = [
        [0, -13, 29],
        [13, 0, -25],
        [-29, 25, 0]
    ]

    mono_violation_game_2 = [
        [0, -15, 13],
        [15, 0, -9],
        [-13, 9, 0]
    ]

    mono_violation_game_3 = [
        [0, -19, 3],
        [19, 0, -17],
        [-3, 17, 0]
    ]


    mono_violation_game_4 = [
        [0, -15, 5],
        [15, 0, -20],
        [-5, 20, 0]
    ]

    a_vs_b = -5

    test_game = [
        [0, a_vs_b, 3],
        [-a_vs_b, 0, -7],
        [-3, 7, 0]
    ]

    # Semi-mono violation?
    # [[  0   1  -1  13]
    #  [ -1   0 -19   7]
    #  [  1  19   0  -7]
    #  [-13  -7   7   0]]

    # Ditto
    # [[  0  13 -27 -23]
    #  [-13   0  -7  11]
    #  [ 27   7   0 -23]
    #  [ 23 -11  23   0]]

    # Ditto
    # [[  0  17 -29  -9]
    #  [-17   0  25   9]
    #  [ 29 -25   0 -13]
    #  [  9  -9  13   0]]

    semi_mono_violation = [
        [0, 15, 13, -19],
        [-15, 0, 21, -27],
        [-13, -21, 0, 15],
        [19, 27, -15, 0]
    ]

    # Counterexample for iterated removal of bad actions
    # [[  0  -3  27 -15]
    #  [  3   0   3  -1]
    #  [-27  -3   0  11]
    #  [ 15   1 -11   0]]

    # Counterexample for always winner of C2-ML
    # [[  0  19 -23 -23]
    #  [-19   0  21  25]
    #  [ 23 -21   0 -15]
    #  [ 23 -25  15   0]]


    # Current test
    # [[  0 -27  23]
    #  [ 27   0 -11]
    #  [-23  11   0]]

    # m_da smallest edge, but d wins
    # [[  0   9  17  -3]
    #  [ -9   0  25   5]
    #  [-17 -25   0   7]
    #  [  3  -5  -7   0]]

    simple_test_game = [
        [0, 1, 7, -11],
        [-1, 0, 3, 9],
        [-7, -3, 0, 5],
        [11, -9, -5, 0]
    ]

    # Those two games are interesting because they are identical except for the value of the largest margin,
    # but seem to converge to different winners.
    same_order_different_winner_0 = [
        [0, 11, 7, -9],
        [-11, 0, 1, 5],
        [-7, -1, 0, 3],
        [9, -5, -3, 0]
    ]

    same_order_different_winner_1 = [
        [0, 13, 7, -9],
        [-13, 0, 1, 5],
        [-7, -1, 0, 3],
        [9, -5, -3, 0]
    ]

    slow_convergence = [
        [0, 25, 17, -27],
        [-25, 0, 19, 29],
        [-17, -19, 0, 5],
        [27, -29, -5, 0]
    ]

    no_slack_game = [
        [0, 15, 3, -9],
        [-15, 0, 1, 25],
        [-3, -1, 0, 5],
        [9, -25, -5, 0]
    ]

    # jumping_at_power_8 = [
    #     [0, -3, 11, -17],
    #     [3, 0, -49, 11],
    #     [-11, 49, 0, 1],
    #     [17, -11, -1, 0]
    # ]

    jumping_at_power_8 = [
        [0, 49, 1, -11],
        [-49, 0, 11, 3],
        [-1, -11, 0, 17],
        [11, -3, -17, 0]
    ]

    # game with 6 alternatives and only one slacking inequality
    # [[0 - 7  15   1  15  15]
    #  [7   0   9   3 - 7 - 29]
    # [-15 - 9
    # 0
    # 13 - 5
    # 15]
    # [-1 - 3 - 13   0 - 15  17]
    # [-15
    # 7
    # 5
    # 15
    # 0 - 11]
    # [-15  29 - 15 - 17  11   0]]

    # Weird game with only one single slacking inequality
    # [[0  25 - 3 - 23 - 15 - 9]
    #  [-25   0  27 - 21   1 - 11]
    # [3 - 27
    # 0
    # 5
    # 7
    # 17]
    # [23  21 - 5   0  29 - 19]
    # [15 - 1 - 7 - 29
    # 0 - 13]
    # [9  11 - 17  19  13   0]]

    # Changing number of slacking vars
    slack_var_count_change = [
        [0, 9, 1, -27, 19, -21],
        [-9, 0, -5, 3, -17, -25],
        [-1, 5, 0, -15, 7, 23],
        [27, -3, 15, 0, -13, 29],
        [-19, 17, -7, 13, 0, -11],
        [21, 25, -23, -29, 11, 0]
    ]


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
    #     alternative_count=4,
    #     max_margin=11,
    #     ensure_uniqueness=True,
    #     ensure_oddity=True,
    #     exclude_weak_condorcet_winner=True,
    #     exclude_weak_condorcet_loser=True
    # )
    game = create_graph_specific_ordered_wmg([(2, 3), (0, 2), (0, 1), (3, 0), (1, 2), (1, 3)], max_margin=11)
    print(game)
    power_sequence(game, print_output=True, print_slack=True)

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

    # mass_winner_slacker_combination_check(game_count=1000, print_output=True)
