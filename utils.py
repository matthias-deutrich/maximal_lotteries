import numpy as np
from scipy.optimize import linprog
from sympy import Matrix, zeros, ones, Symbol, oo, pprint
from sympy.functions.elementary.complexes import Abs, sign

superscript = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


class SolverException(Exception):
    pass


# TODO check if there is some way to use simplification while changing positive to nonnegative
def make_variable(name, force_positive=True):
    return Symbol(name, positive=force_positive, integer=True)


t = make_variable('t')


def maximin(game_matrix: np.ndarray):
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

        return security_level, strategy, result.slack
    else:
        raise SolverException(f'The problem could not be solved with the following reason: {result.message}')


def maximal_lottery(matrix: Matrix):
    security_level, lottery, slack = maximin(sympy_to_numpy(matrix))
    security_level = round(security_level, 3)
    if security_level != 0:
        raise SolverException(f'The security level was {security_level} instead of 0, so something went wrong.')
    lottery = [round(prob, 3) for prob in lottery]
    slack = [round(prob, 3) for prob in slack]
    return lottery, slack


def t_matrix(matrix: Matrix) -> Matrix:
    matrix = matrix.copy()
    for i in range(len(matrix)):
        matrix[i] = 0 if matrix == 0 else Abs(matrix[i])**t * sign(matrix[i])
    # tmp = Symbol('tmp')
    # for s in matrix.free_symbols:
    #     matrix = matrix.subs(s, tmp)
    #     matrix = matrix.subs(-tmp, -s**t)
    #     matrix = matrix.subs(tmp, s**t)
    return matrix


def sub_matrix(matrix: Matrix, indices: set[int] = None) -> Matrix:
    matrix = matrix.copy()
    if not indices:
        return matrix
    indices_delete = [i for i in range(matrix.rows-1, -1, -1) if i not in indices]
    for i in indices_delete:
        matrix.row_del(i),
        matrix.col_del(i)
    return matrix


def matrix_to_linalg(matrix: Matrix) -> Matrix:
    matrix = matrix.col_insert(matrix.cols, zeros(matrix.rows, 1))
    return matrix.row_insert(matrix.rows, ones(1, matrix.cols))


# def pretty_print(matrix: Matrix):
#     return '\n'.join([str(list(matrix[i, :])) for i in range(matrix.rows)])


def sympy_to_numpy(matrix: Matrix):
    return np.array(matrix).astype(int)


def has_condorcet_case(matrix: Matrix):
    for sym in matrix.free_symbols:
        matrix = matrix.subs(sym, 1)
    return (matrix * Matrix.ones(matrix.rows, 1)).norm(ord=oo) == matrix.rows - 1


def support_set(lottery):
    return {i for (i, p) in enumerate(lottery) if p > 0}


def print_rref(arg: (Matrix, list)):
    if arg[1]:
        print(f'Assuming supp(ML_t) = {set(range(arg[0].rows + len(arg[1]) - 1)) - {i for i, _ in arg[1]} }')
    else:
        print('Assuming full support:')
    pprint(arg[0])
    for index, inequality in arg[1]:
        print(f'Inequality {index}:')
        pprint(inequality)


# def detect_slack_change(slacks):
#     positive_slacks = np.greater_equal(slacks, 0.001)
#     slacking_vars
#
#     def all_slacking_alts(slacks):
#         positive_slacks = np.greater_equal(slacks, 0.001)
#         positive_vars = np.any(positive_slacks, axis=0)
#         return set(np.where(positive_vars)[0])
