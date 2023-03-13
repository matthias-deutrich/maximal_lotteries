from sympy import Matrix, pprint, init_printing, simplify

from sample_games import InterestingGames
from utils import *
from game_matrices import GameMatrix, s, l
from timeit import default_timer as timer


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
        curr_game.subs(s, 21)
        curr_game.subs(l, 29)
        if curr_game.detect_support_changes_numeric(print_output=True):
            pprint(curr_game.matrix)
            print(curr_game.matrix)
            return True
    return False

def time_solutions(game, support, t_val):
    # INVERSION TIMES (one sample):
    # ADJ: 81
    # BLOCK: 46
    # CH: 46
    # GE: did not finish
    # LDL: 46
    # LU: 46
    # QR: did not finish
    start = timer()
    left_mpi_solution = game.solve_assuming_support(support, mode='left_mpi',
                                                    inversion_method='ADJ', print_output=False)[0]
    end = timer()
    print(f'ADJ took {end - start}')
    pprint(left_mpi_solution.subs(t, t_val))
    pprint(left_mpi_solution.applyfunc(simplify))

    start = timer()
    left_mpi_solution = game.solve_assuming_support(support, mode='left_mpi',
                                                    inversion_method='BLOCK', print_output=False)[0]
    end = timer()
    print(f'BLOCK took {end - start}')
    pprint(left_mpi_solution.subs(t, t_val))
    # pprint(left_mpi_solution.applyfunc(simplify))

    start = timer()
    left_mpi_solution = game.solve_assuming_support(support, mode='left_mpi',
                                                    inversion_method='CH', print_output=False)[0]
    end = timer()
    print(f'CH took {end - start}')
    pprint(left_mpi_solution.subs(t, t_val))
    # pprint(left_mpi_solution.applyfunc(simplify))

    # start = timer()
    # left_mpi_solution = game.solve_assuming_support(support, mode='left_mpi',
    #                                                 inversion_method='GE', print_output=False)[0]
    # end = timer()
    # print(f'GE took {end - start}')
    # pprint(left_mpi_solution.subs(t, t_val))

    start = timer()
    left_mpi_solution = game.solve_assuming_support(support, mode='left_mpi',
                                                    inversion_method='LDL', print_output=False)[0]
    end = timer()
    print(f'LDL took {end - start}')
    pprint(left_mpi_solution.subs(t, t_val))
    # pprint(left_mpi_solution.applyfunc(simplify))

    start = timer()
    left_mpi_solution = game.solve_assuming_support(support, mode='left_mpi',
                                                    inversion_method='LU', print_output=False)[0]
    end = timer()
    print(f'LU took {end - start}')
    pprint(left_mpi_solution.subs(t, t_val))
    # pprint(left_mpi_solution.applyfunc(simplify))

    # start = timer()
    # left_mpi_solution = game.solve_assuming_support(support, mode='left_mpi',
    #                                                 inversion_method='QR', print_output=False)[0]
    # end = timer()
    # print(f'QR took {end - start}')
    # pprint(left_mpi_solution.subs(t, t_val))


if __name__ == '__main__':
    # pprint(InterestingGames.generic_chris_example.rref())

    # game = GameMatrix.random_matrix(5, exclude_weak_condorcet_winner=True,
    #                                 exclude_weak_condorcet_loser=True, exclude_zero=True)
    game = InterestingGames.presentation_game_chosen
    #
    # # game = InterestingGames.dichotomous_support_change
    # game.force_positive_symbols()
    # game.solve_assuming_support({0, 1, 3}, print_output=True)
    # game.solve_assuming_support({0, 1, 4}, print_output=True)
    # game_rref = game.rref({0, 1, 4}, add_validity_condition=False)[0]
    # pprint(game_rref)
    # game_rref_coeff = game_rref.row_insert(game_rref.rows, Matrix.ones(1, game_rref.cols))
    # print('Rank of coefficient matrix: ', game_rref_coeff.rank())
    # rhs = Matrix.zeros(game_rref_coeff.rows, 1)
    # rhs[-1] = 1
    # game_rref_augmented = game_rref_coeff.col_insert(game_rref_coeff.cols, rhs)
    # print('Rank of augmented matrix: ', game_rref_augmented.rank())
    # # print_rref(game_rref)
    # game.subs(s, 25)
    # game.subs(l, 29)




    print(game)
    game.power_sequence(print_output=True, print_slack=True)

    # support = {0, 2, 4}
    # support = {1, 3, 4}
    support = None
    t_val = 2
    # time_solutions(game, support,t_val)
    # game.fix_t(t_val)

    main_solution, main_inequalities = game.solve_assuming_support(support, print_output=True)
    pprint(main_solution.subs(t, 0))
    pprint(main_solution.subs(t, 1))
    pprint(main_solution.subs(t, 2))
    pprint(main_solution.subs(t, 3))
    # left_mpi_solution = game.solve_assuming_support(support, mode='left_mpi', print_output=True)[0]
    # pprint(rref_solution.subs(t, t_val))



    # general_mpi_solution = game.solve_assuming_support(support, mode='general_mpi', print_output=True)[0]

    # pprint(rref_solution.subs(t, t_val))
    # pprint(mpi_solution.subs(t, t_val))

    # game.solve_assuming_support(print_output=True)
    # game.solve_assuming_support({1, 3, 4}, print_output=True)

    # game = InterestingGames.generic_four_game
    # game.force_positive_symbols()
    # visual_ledermann_proof(game.t_matrix())

    # game = InterestingGames.dichotomous_support_change_7alts
    # game.solve_assuming_support({1, 5, 6}, print_output=True)
    # game.solve_assuming_support({0, 1, 3, 5, 6}, print_output=True)
    # game.subs(s, 25)
    # game.subs(l, 29)
    # game.power_sequence(print_output=True, print_slack=True)

    # pprint(game.rref())
    # pprint(game.matrix)
    # game.detect_support_changes_numeric(print_output=True)

    # game = InterestingGames.two_slack_changes
    # # game.solve_assuming_support(print_output=True)
    # game.solve_assuming_support({2, 3, 4}, print_output=True)
    # game.solve_assuming_support({0, 2, 4}, print_output=True)
    # game.power_sequence(print_output=True, print_slack=True)

    # tmp = Matrix([
    #     [0, s**t, s**t, -l**t],
    #     [-s**t, 0, l**t, -l**t],
    #     [-s**t, -l**t, 0, s**t],
    #     [l**t, l**t, -s**t, 0]
    # ])
    # tmp = Matrix([
    #     [0, s ** t, s ** t, s ** t],
    #     [-s ** t, 0, s ** t, l ** t],
    #     [-s ** t, -s ** t, 0, s ** t],
    #     [-s ** t, -l ** t, -s ** t, 0]
    # ])
    # pprint(tmp.rref()[0])
    # tmp2 = tmp.subs(s, 1).subs(l, 2)
    # pprint(tmp2.rref()[0])
    # tmp3 = tmp2.subs(t, 1)
    # pprint(tmp3)
    # pprint(tmp3.rref()[0])


    # find_support_changing_game(5)

    # InterestingGames.chris_example.power_sequence(print_output=True, print_slack=True)
