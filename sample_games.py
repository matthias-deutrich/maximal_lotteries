from game_matrices import GameMatrix, s, l
from sympy import Matrix
from utils import make_variable


class InterestingGames:
    sample_game = GameMatrix([
        [0, 1, -3, -3],
        [-1, 0, 1, 11],
        [3, -1, 0, -3],
        [3, -11, 3, 0]
    ])

    interesting_game_1 = GameMatrix([
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
    non_mono_at_3 = GameMatrix([
        [0, -41, 45, 17],
        [41, 0, -29, 21],
        [-45, 29, 0, 17],
        [-17, -21, -17, 0]
    ])

    # Ditto, for three alternatives
    mono_violation_game_1 = GameMatrix([
        [0, -13, 29],
        [13, 0, -25],
        [-29, 25, 0]
    ])

    mono_violation_game_2 = GameMatrix([
        [0, -15, 13],
        [15, 0, -9],
        [-13, 9, 0]
    ])

    mono_violation_game_3 = GameMatrix([
        [0, -19, 3],
        [19, 0, -17],
        [-3, 17, 0]
    ])


    mono_violation_game_4 = GameMatrix([
        [0, -15, 5],
        [15, 0, -20],
        [-5, 20, 0]
    ])

    a_vs_b = -5

    test_game = GameMatrix([
        [0, a_vs_b, 3],
        [-a_vs_b, 0, -7],
        [-3, 7, 0]
    ])

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

    semi_mono_violation = GameMatrix([
        [0, 15, 13, -19],
        [-15, 0, 21, -27],
        [-13, -21, 0, 15],
        [19, 27, -15, 0]
    ])

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

    simple_test_game = GameMatrix([
        [0, 1, 7, -11],
        [-1, 0, 3, 9],
        [-7, -3, 0, 5],
        [11, -9, -5, 0]
    ])

    # Those two games are interesting because they are identical except for the value of the largest margin,
    # but seem to converge to different winners.
    same_order_different_winner_0 = GameMatrix([
        [0, 11, 7, -9],
        [-11, 0, 1, 5],
        [-7, -1, 0, 3],
        [9, -5, -3, 0]
    ])

    same_order_different_winner_1 = GameMatrix([
        [0, 13, 7, -9],
        [-13, 0, 1, 5],
        [-7, -1, 0, 3],
        [9, -5, -3, 0]
    ])

    slow_convergence = GameMatrix([
        [0, 25, 17, -27],
        [-25, 0, 19, 29],
        [-17, -19, 0, 5],
        [27, -29, -5, 0]
    ])

    no_slack_game = GameMatrix([
        [0, 15, 3, -9],
        [-15, 0, 1, 25],
        [-3, -1, 0, 5],
        [9, -25, -5, 0]
    ])

    # jumping_at_power_8 = [
    #     [0, -3, 11, -17],
    #     [3, 0, -49, 11],
    #     [-11, 49, 0, 1],
    #     [17, -11, -1, 0]
    # ]

    jumping_at_power_8 = GameMatrix([
        [0, 49, 1, -11],
        [-49, 0, 11, 3],
        [-1, -11, 0, 17],
        [11, -3, -17, 0]
    ])

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
    slack_var_count_change = GameMatrix([
        [0, 9, 1, -27, 19, -21],
        [-9, 0, -5, 3, -17, -25],
        [-1, 5, 0, -15, 7, 23],
        [27, -3, 15, 0, -13, 29],
        [-19, 17, -7, 13, 0, -11],
        [21, 25, -23, -29, 11, 0]
    ])

    slacker_supposedly_3 = GameMatrix([
        [0, 7, 1, -3],
        [-7, 0, 19, 29],
        [-1, -19, 0, 11],
        [3, -29, -11, 0]
    ])

    # Seems like numerical errors
    slacker_supposedly_3_v2 = GameMatrix([
        [0, 13, 1, -23],
        [-13, 0, 3, 27],
        [-1, -3, 0, 11],
        [23, -27, -11, 0]
    ])

    # Five alternatives, no slackers
    five_alts_no_slackers = GameMatrix([
        [0, 5, -3, -21, 43],
        [-5, 0, 35, -11, -45],
        [3, -35, 0, 27, -25],
        [21, 11, -27, 0, -17],
        [-43, 45, 25, 17, 0]
    ])

    # Another one
    # [[0 - 15  47 - 39 - 9]
    #  [15   0 - 25  43 - 23]
    # [-47
    # 25
    # 0
    # 13 - 21]
    # [39 - 43 - 13   0  41]
    # [9
    # 23
    # 21 - 41
    # 0]]

    simple_example_game = GameMatrix([
        [0, 3, -1],
        [-3, 0, 1],
        [1, -1, 0]
    ])

    outdated_efficiency_counterexample = GameMatrix([
        [0, 1, 1, -1, -1, 1, 1, 1, 0, -2],
        [-1, 0, 1, 1, -1, -2, 1, 1, 1, 0],
        [-1, -1, 0, 1, 1, 0, -2, 1, 1, 1],
        [1, -1, -1, 0, 1, 1, 0, -2, 1, 1],
        [1, 1, -1, -1, 0, 1, 1, 0, -2, 1],
        [-1, 2, 0, -1, -1, 0, 2, 0, 0, -2],
        [-1, -1, 2, 0, -1, -2, 0, 2, 0, 0],
        [-1, -1, -1, 2, 0, 0, -2, 0, 2, 0],
        [0, -1, -1, -1, 2, 0, 0, -2, 0, 2],
        [2, 0, -1, -1, -1, 2, 0, 0, -2, 0]
    ])

    efficiency_counterexample = GameMatrix([
        [0, 0, 0, 2, 2, -3],
        [0, 0, 0, -3, 2, 2],
        [0, 0, 0, 2, -3, 2],
        [-2, 3, -2, 0, 4, -4],
        [-2, -2, 3, -4, 0, 4],
        [3, -2, -2, 4, -4, 0]
    ])

    # [[0   5  21  11 - 27]
    #  [-5   0  29 - 23 - 7]
    # [-21 - 29
    # 0
    # 23 - 7]
    # [-11  23 - 23   0  11]
    # [27
    # 7
    # 7 - 11
    # 0]]

    two_slack_changes = GameMatrix([
        [0, -7, 7, -13, -7],
        [7, 0, 23, -15, -25],
        [-7, -23, 0, -5, 3],
        [13, 15, 5, 0, -7],
        [7, 25, -3, 7, 0]
    ])

    chris_example = GameMatrix([
        [0, 3, -1, -1, -1],
        [-3, 0, 3, 1, 3],
        [1, -3, 0, -3, 1],
        [1, -1, 3, 0, -3],
        [1, -3, -1, 3, 0]
    ])

    generic_chris_example = GameMatrix(
        Matrix([
            [0, l, -s, -s, -s],
            [-l, 0, l, s, l],
            [s, -l, 0, -l, s],
            [s, -s, l, 0, -l],
            [s, -l, -s, l, 0]
        ])
    )

    generic_five_example_2 = GameMatrix(
        Matrix([
            [0, l, -l, -s, -s],
            [-l, 0, l, s, l],
            [l, -l, 0, s, s],
            [s, -s, -s, 0, l],
            [s, -l, -s, -l, 0]
        ])
    )

    a = make_variable('a', force_positive=False)
    b = make_variable('b', force_positive=False)
    c = make_variable('c', force_positive=False)
    d = make_variable('d', force_positive=False)
    e = make_variable('e', force_positive=False)
    f = make_variable('f', force_positive=False)
    g = make_variable('g', force_positive=False)
    h = make_variable('h', force_positive=False)
    i = make_variable('i', force_positive=False)
    j = make_variable('j', force_positive=False)
    k = make_variable('k', force_positive=False)
    l = make_variable('l', force_positive=False)
    m = make_variable('m', force_positive=False)
    n = make_variable('n', force_positive=False)
    o = make_variable('o', force_positive=False)

    generic_four_game = GameMatrix(
        Matrix([
            [0, a, d, f],
            [-a, 0, b, e],
            [-d, -b, 0, c],
            [-f, -e, -c, 0]
        ])
    )

    generic_five_game = GameMatrix(
        Matrix([
            [0, a, e, h, j],
            [-a, 0, b, f, i],
            [-e, -b, 0, c, g],
            [-h, -f, -c, 0, d],
            [-j, -i, -g, -d, 0]
        ])
    )

    generic_six_game = GameMatrix(
        Matrix([
            [0, a, f, j, m, o],
            [-a, 0, b, g, k, n],
            [-f, -b, 0, c, h, l],
            [-j, -g, -c, 0, d, i],
            [-m, -k, -h, -d, 0, e],
            [-o, -n, -l, -i, -e, 0]
        ])
    )

    # semi_generic_five_game_4_alts = generic_five_game.subs()
    dichotomous_support_change_7alts = GameMatrix(
        Matrix([
            [0, -l, l, s, -l, s, -s],
            [l, 0, l, s, s, -l, s],
            [-l, -l, 0, l, l, s, -l],
            [-s, -s, -l, 0, l, -l, l],
            [l, -s, -l, -l, 0, s, -s],
            [-s, l, -s, l, -s, 0, -l],
            [s, -s, l, -l, s, l, 0]
        ])
    )
    # ⎡ 0   -29  29   21   -29  21   -21⎤
    # ⎢                                 ⎥
    # ⎢29    0   29   21   21   -29  21 ⎥
    # ⎢                                 ⎥
    # ⎢-29  -29   0   29   29   21   -29⎥
    # ⎢                                 ⎥
    # ⎢-21  -21  -29   0   29   -29  29 ⎥
    # ⎢                                 ⎥
    # ⎢29   -21  -29  -29   0   21   -21⎥
    # ⎢                                 ⎥
    # ⎢-21  29   -21  29   -21   0   -29⎥
    # ⎢                                 ⎥
    # ⎣21   -21  29   -29  21   29    0 ⎦

    dichotomous_support_change = GameMatrix(
        Matrix([
            [0, s, s, -s, -l],
            [-s, 0, l, l, l],
            [-s, -l, 0, -s, l],
            [s, -l, s, 0, s],
            [l, -l, -l, -s, 0]
        ])
    )

    wat = GameMatrix(
        Matrix([[0, -s, -s, -s, s], [s, 0, l, -s, l], [s, -l, 0, -s, -l], [s, s, s, 0, -s], [-s, -l, l, s, 0]]))
    
    dichotomous_support_size_change = GameMatrix(
        Matrix([
            [0, s, -s, -s, -l],
            [-s, 0, l, l, s],
            [s, -l, 0, s, -l],
            [s, -l, -s, 0, l],
            [l, -s, l, -l, 0]
        ])
    )

    presentation_game = GameMatrix(
        Matrix([
            [0, -8, -9, 3, -9, -26, -30],
            [8, 0, 21, -14, 20, -20, -2],
            [9, -21, 0, -21, -11, -6, 29],
            [-3, 14, 21, 0, 7, -28, 30],
            [9, -20, 11, -7, 0, 30, -15],
            [26, 20, 6, 28, -30, 0, 30],
            [30, 2, -29, -30, 15, -30, 0]
        ])
    )

    presentation_game_2 = GameMatrix(
        Matrix([
            [0, 19, -29, 30, 15, -6, 29],
            [-19, 0, 27, -27, 1, -16, 1],
            [29, -27, 0, -28, 7, -14, -10],
            [-30, 27, 28, 0, 8, -16, -3],
            [-15, -1, -7, -8, 0, 10, 27],
            [6, 16, 14, 16, -10, 0, 20],
            [-29, -1, 10, 3, -27, -20, 0]
        ])
    )

    presentation_game_3 = GameMatrix(
        Matrix([
            [0, 23, -28, -28, -23, -12, -18],
            [-23, 0, 14, 21, -14, 11, -17],
            [28, -14, 0, 28, 13, -21, -25],
            [28, -21, -28, 0, 3, 28, 4],
            [23, 14, -13, -3, 0, -22, 22],
            [12, -11, 21, -28, 22, 0, 16],
            [18, 17, 25, -4, -22, -16, 0]
        ])
    )

    # Matrix([[0, -22, 23, -15, 22], [22, 0, 20, 27, -4], [-23, -20, 0, 8, 12], [15, -27, -8, 0, 29], [-22, 4, -12, -29, 0]])

    # Matrix([[0, 22, 18, -26, 15], [-22, 0, 3, 3, -16], [-18, -3, 0, 20, -11], [26, -3, -20, 0, -13], [-15, 16, 11, 13, 0]])

    presentation_game_4 = GameMatrix(
        Matrix([
            [0, 24, -17, -23, -22],
            [-24, 0, -11, 3, 21],
            [17, 11, 0, 18, -22],
            [23, -3, -18, 0, 9],
            [22, -21, 22, -9, 0]
        ])
    )

    presentation_game_5 = GameMatrix(
        Matrix([
            [0, -16, 2, 16, 15],
            [16, 0, 27, -14, -20],
            [-2, -27, 0, -14, 27],
            [-16, 14, 14, 0, 12],
            [-15, 20, -27, -12, 0]
        ])
    )

    presentation_game_chosen = GameMatrix(
        Matrix([
            [0, 2, 6, 5, -11],
            [-2, 0, 20, 6, -11],
            [-6, -20, 0, 10, 16],
            [-5, -6, -10, 0, 12],
            [11, 11, -16, -12, 0]
        ])
    )

