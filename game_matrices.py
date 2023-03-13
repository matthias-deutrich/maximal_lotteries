from random import randrange, shuffle, choice
from sympy import simplify, expand, cancel, powsimp, factor, gcd
from utils import *

s = make_variable('s')
l = make_variable('l')


class GameMatrix:
    def __init__(self, matrix):
        self.matrix = Matrix(matrix)
        self.n = self.matrix.rows
        if self.n != self.matrix.cols:
            raise ValueError(f'Matrix must be square! Received matrix was {self.matrix}')

        # Check if the matrix is actually skew-symmetric
        for i in range(self.n):
            for j in range(i + 1):
                if self.matrix[i, j] != -self.matrix[j, i]:
                    raise ValueError(f'Matrix must be skew-symmetric! Received matrix was {self.matrix}')

        self.free_symbols = self.matrix.free_symbols

        self.fixed_t = None
        # for s in self.free_symbols:
        #     if not s.is_nonnegative:
        #         raise ValueError(f'All variables used in the matrix must be declared to be non-negative.')

    @classmethod
    def random_matrix(
            cls,
            alternative_count,
            max_margin=30,
            ensure_uniqueness=False,
            ensure_oddity=False,
            exclude_weak_condorcet_winner=False,
            exclude_weak_condorcet_loser=False,
            exclude_zero=False
    ):
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
                    return GameMatrix.random_matrix(
                        alternative_count=alternative_count,
                        max_margin=max_margin,
                        ensure_uniqueness=ensure_uniqueness,
                        ensure_oddity=ensure_oddity,
                        exclude_weak_condorcet_winner=exclude_weak_condorcet_winner,
                        exclude_weak_condorcet_loser=exclude_weak_condorcet_loser,
                        exclude_zero=exclude_zero
                    )
        if exclude_weak_condorcet_loser:
            for i in range(alternative_count):
                if np.logical_and.reduce(weighted_majority_graph[i, :] <= 0):
                    return GameMatrix.random_matrix(
                        alternative_count=alternative_count,
                        max_margin=max_margin,
                        ensure_uniqueness=ensure_uniqueness,
                        ensure_oddity=ensure_oddity,
                        exclude_weak_condorcet_winner=exclude_weak_condorcet_winner,
                        exclude_weak_condorcet_loser=exclude_weak_condorcet_loser,
                        exclude_zero=exclude_zero
                    )
        return GameMatrix(weighted_majority_graph)

    @classmethod
    def random_dichotomous_matrix(cls, alternatives, exclude_condorcet_cases=True):
        matrix = Matrix.zeros(alternatives)
        for i in range(alternatives):
            for j in range(i + 1, alternatives):
                matrix[i, j] = choice([s, l, -s, -l])
                matrix[j, i] = -matrix[i, j]

        if exclude_condorcet_cases:
            if has_condorcet_case(matrix):
                return GameMatrix.random_dichotomous_matrix(alternatives=alternatives, exclude_condorcet_cases=True)

        return GameMatrix(matrix)

    def __str__(self):
        return str(self.matrix)

    def as_numpy(self):
        return sympy_to_numpy(self.matrix)

    def subs(self, args, kwargs):
        self.matrix = self.matrix.subs(args, kwargs)

    def fix_t(self, fixed_t: int):
        self.fixed_t = fixed_t

    def t_matrix(self):
        return t_matrix(self.matrix, fixed_t=self.fixed_t)

    def force_positive_symbols(self):
        tmp = make_variable('tmp', force_positive=True)
        for sym in self.matrix.free_symbols:
            self.subs(sym, tmp)
            self.subs(tmp, make_variable(sym.name, force_positive=True))

    def power_sequence(
            self,
            max_power=10,
            print_output=False,
            print_slack=False
    ):
        lotteries, slacks = [], []
        matrix = t_matrix(self.matrix)
        for power in range(max_power + 1):
            try:
                curr_matrix = sympy_to_numpy(matrix.subs(t, power))
                lottery, slack = maximal_lottery(curr_matrix)
                lotteries.append(lottery)
                slacks.append(slack)
                if print_output:
                    print(f'For \u03C4(x) = x{"" if power == 1 else str(power).translate(superscript)}, '
                          f'a maximal lottery is {lottery}' + (f' with slack {slack}' if print_slack else ''))
            except SolverException:
                break
        lotteries = np.asarray(lotteries)
        slacks = np.asarray(slacks)
        return lotteries, slacks

    def detect_support_changes_numeric(self, max_power=6, print_output=False):
        lotteries, _ = self.power_sequence(max_power=max_power)
        supports = [support_set(lottery) for lottery in lotteries]
        for power, (s1, s2) in enumerate(zip(supports, supports[1:])):
            if s1 != s2:
                if print_output:
                    print(f'The support of this game changes between powers {power} and {power + 1}, '
                          f'with the original support having size {len(s1)} and the new one size {len(s2)}.')
                return True
        return False

    def rref(self, indices: set[int] = None, add_validity_condition=True):
        matrix = sub_matrix(self.t_matrix(), indices)
        if add_validity_condition:
            matrix = matrix_to_linalg(matrix)
        matrix_rref = matrix.rref()[0]
        return (matrix_rref,
                [(j, t_matrix(self.matrix)[:, j].transpose()) for j in range(self.n) if j not in indices]
                if indices else [])

    def general_mpi(self, indices: set[int] = None, add_validity_condition=True):
        matrix = sub_matrix(self.t_matrix(), indices)
        if add_validity_condition:
            matrix = matrix_to_linalg(matrix)
            matrix = matrix[:, :-1]
        return matrix.pinv()

    def left_mpi(self, indices: set[int] = None, add_validity_condition=True, inversion_method='ADJ'):
        matrix = sub_matrix(self.t_matrix(), indices)
        if add_validity_condition:
            matrix = matrix_to_linalg(matrix)
            matrix = matrix[:, :-1]

        tmp = matrix.transpose() * matrix

        if inversion_method == 'ADJ':
            inv = tmp.inverse_ADJ()
        elif inversion_method == 'BLOCK':
            inv = tmp.inverse_BLOCK()
        elif inversion_method == 'CH':
            inv = tmp.inverse_CH()
        elif inversion_method == 'GE':
            inv = tmp.inverse_GE()
        elif inversion_method == 'LDL':
            inv = tmp.inverse_LDL()
        elif inversion_method == 'LU':
            inv = tmp.inverse_LU()
        elif inversion_method == 'QR':
            inv = tmp.inverse_QR()
        else:
            inv = tmp.inv()

        solution = inv * matrix.transpose()
        return solution.applyfunc(simplify) if inversion_method == 'ADJ' else solution
        # return (matrix.transpose() * matrix).inverse_LDL() * matrix.transpose()

    # def mpi_solution(self, indices: set[int] = None):
    #     support_solution = self.mpi(indices=indices, add_validity_condition=True)
    #     if indices:
    #         solution = Matrix.zeros(self.n, 1)
    #         for i, k in enumerate(sorted(indices)):
    #             solution[k] = support_solution[i, -1]
    #     else:
    #         solution = support_solution[:, -1]
    #     return solution.transpose()

    def solve_assuming_support(self, indices: set[int] = None, mode='rref', inversion_method='ADJ',
                               use_custom_simplification=True, print_output=False):
        if (len(indices) if indices else self.n) % 2 == 0:
            raise SolverException('The size of the matrix must be odd.')

        if mode == 'general_mpi':
            matrix_solution = self.general_mpi(indices=indices, add_validity_condition=True)
        elif mode == 'left_mpi':
            matrix_solution = self.left_mpi(indices=indices,
                                            inversion_method=inversion_method,
                                            add_validity_condition=True)
        else:
            mode = 'rref'
            rref, _ = self.rref(indices=indices, add_validity_condition=True)
            if rref[-1] != 0:
                raise SolverException('The rank of the matrix was full, which should not be possible.')
            if rref[-2, -2] != 1:
                raise SolverException('The rank of the matrix was less than n - 1, which should not be possible.')
            matrix_solution = rref[:-1, :]

        if indices:
            solution = Matrix.zeros(self.n, 1)
            for i, k in enumerate(sorted(indices)):
                solution[k] = matrix_solution[i, -1]
        else:
            solution = matrix_solution[:, -1]
        solution = solution.transpose()

        inequalities = [(solution * self.t_matrix()[:, i])[0] for i in range(self.n) if i not in indices]\
            if indices else []

        if use_custom_simplification:
            solution = solution.applyfunc(custom_simplify)
            inequalities = [custom_simplify(cancel(ineq)) for ineq in inequalities]

        inequalities = [ineq >= 0 for ineq in inequalities]

        if print_output:
            if indices:
                print(f'Assuming supp(ML_t) = {indices}, a solution (computed in {mode} mode) is:')
                pprint(solution)
                print('To be valid, this solution must fulfil the following inequalities:')

                for ineq in inequalities:
                    pprint(ineq)
            else:
                print(f'Assuming full support, a solution (computed in {mode} mode) is:')
                pprint(solution)

        return solution, inequalities

