from sympy import Matrix, pprint, init_printing

from sample_games import InterestingGames
from utils import *
from game_matrices import GameMatrix, s, l


# color_dict = {
#     0: '\033[94m',
#     1: '\033[96m',
#     2: '\033[92m',
#     3: '\033[93m',
#     -1: '\033[0m'
# }

#
# def winner(lotteries, threshold=0.55, print_output=False):
#     winning_alternative = np.argmax(lotteries, 1)[-1]
#     winning_prob = np.max(lotteries, 1)[-1]
#     if threshold:
#         if winning_prob < threshold:
#             raise CustomException(f'The winning alternative in the last lottery was only {winning_prob}')
#     if print_output:
#         print(f'The last lottery was won by alternative {winning_alternative} '
#               'with sufficient probability to indicate convergence.')
#     return winning_alternative
#
#
# def all_slacking_alts(slacks):
#     positive_slacks = np.greater_equal(slacks, 0.001)
#     positive_vars = np.any(positive_slacks, axis=0)
#     return set(np.where(positive_vars)[0])
#
#
# def last_slacking_alt(slacks):
#     positive_slacks = np.greater_equal(slacks[:][-1], 0.001)
#     positive_indices = np.where(positive_slacks)[0]
#     if len(positive_indices) > 1:
#         raise CustomException(f'All equations {positive_indices} had slack, which is more than 1')
#     return positive_indices[0]
#
#
#
# def rref_valid(matrix, indices=None, print_output=False):
#     alt_count = len(matrix[0])
#     rref = rref_from_game(matrix, indices)
#     if print_output:
#         print(rref[0])
#     vals = rref[0][alt_count:(alt_count + 1)**2 - 1:alt_count + 1]
#     is_valid = reduce((lambda x, y: x and y), [val >= 0 for val in vals])
#     return is_valid and rref[0][-1] == 0
#
#
# def always_rank_minus_1_test(n, count=1000):
#     for i in range(count):
#         matrix = create_random_wmg(n, max_margin=30)
#         if np.linalg.matrix_rank(matrix) != n - 1:
#             raise Exception(matrix)

def find_support_changing_game(alternatives, tries=1000):
    for i in range(tries):
        curr_game = GameMatrix.random_dichotomous_matrix(alternatives=alternatives)
        curr_game.subs(s, 1)
        curr_game.subs(l, 29)
        if curr_game.detect_support_changes_numeric():
            pprint(curr_game.matrix)
            print(curr_game.matrix)
            return True
    return False


if __name__ == '__main__':
    # pprint(InterestingGames.generic_chris_example.rref())
    # game = GameMatrix.random_dichotomous_matrix(7)

    game = InterestingGames.dichotomous_support_change
    print_rref(game.rref({0, 1, 3}))
    print_rref(game.rref({0, 1, 4}))
    game.subs(s, 25)
    game.subs(l, 29)
    game.power_sequence(print_output=True, print_slack=True)

    # pprint(game.rref())
    # pprint(game.matrix)
    # game.detect_support_changes_numeric(print_output=True)

    # find_support_changing_game(5)

    # InterestingGames.chris_example.power_sequence(print_output=True, print_slack=True)
