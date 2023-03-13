import math

import numpy as np
from scipy.optimize import linprog
from sympy import Matrix, zeros, ones, Symbol, oo, pprint, powsimp
from sympy.functions.elementary.complexes import Abs, sign
from sympy.core import *
from sympy.ntheory.ecm import ecm

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


def t_matrix(matrix: Matrix, fixed_t=None) -> Matrix:
    used_t = t if fixed_t is None else fixed_t
    matrix = matrix.copy()
    for i in range(len(matrix)):
        matrix[i] = 0 if matrix == 0 else Abs(matrix[i])**used_t * sign(matrix[i])
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


def print_rref(rref: Matrix, inequalities: list = None):
    if inequalities:
        print(f'Assuming supp(ML_t) = {set(range(len(inequalities[0][1]))) - {i for i, _ in inequalities} }')
    else:
        print('Assuming full support:')
    pprint(rref)
    if inequalities:
        for index, inequality in inequalities:
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

def base_gcd(exp):
    bases = []
    factors = exp.args if isinstance(exp, Add) else [exp]
    for s in factors:
        s = s.args[1] if isinstance(s, Mul) and len(s.args) == 2 and isinstance(s.args[0], Integer) else s
        if isinstance(s, Pow):
            if len(s.args) == 2 and s.args[1] == t and isinstance(s.args[0], Integer):
                bases.append(int(s.args[0]))
        else:
            return 1
    return math.gcd(*bases)


def simplify_fraction(numerator, denominator):
    numerator = numerator.powsimp()
    num_gcd = base_gcd(numerator)

    denominator = denominator.powsimp()
    den_gcd = base_gcd(denominator)

    total_gcd = math.gcd(num_gcd, den_gcd)

    return ((numerator / total_gcd ** t).expand().powsimp(combine='base')
            / (denominator / total_gcd ** t).expand().powsimp(combine='base'))
    #
    # tmp = (numerator / total_gcd ** t)
    # tmp2 = (denominator / total_gcd ** t)
    # pprint(tmp)
    # pprint(tmp2)
    #
    # pprint(numerator.powsimp())
    # tmp = numerator + denominator
    # pprint(tmp)
    # print('here')
    # pprint((tmp.powsimp() / (2 ** t)))
    # pprint((tmp.powsimp() / (2 ** t)).expand())
    # pprint(powsimp((tmp.powsimp() / (5 ** t)).expand(), combine='base'))
    # # pprint(expand_power_base((tmp.powsimp() / (2 ** t)).expand()))
    # # pprint((tmp.powsimp() / (2 ** t)).expand().expand_power_base())
    # # pprint((tmp.powsimp() / (2 ** t)).expand().powsimp())
    # # pprint(denominator)
    # return numerator / denominator


def custom_simplify(exp: Expr):
    if isinstance(exp, Mul):
        num_list, den_list = [], []
        for e in exp.args:
            if isinstance(e, Pow) and len(e.args) == 2 and e.args[1] == -1:
                den_list.append(e.args[0])
            else:
                num_list.append(e)
        if num_list and den_list:
            return simplify_fraction(numerator=Mul(*num_list), denominator=Mul(*den_list))

    return exp
