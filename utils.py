import numpy as np
from scipy.optimize import linprog
from sympy import Matrix, Symbol, zeros, ones


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


def t_matrix(matrix: Matrix) -> Matrix:
    matrix = matrix.copy()
    t = Symbol('t')
    for i in range(len(matrix)):
        matrix[i] = 0 if matrix[i] == 0 else matrix[i] ** t if matrix[i] > 0 else -(-matrix[i]) ** t
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
