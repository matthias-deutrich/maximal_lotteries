from random import randrange, shuffle, choice

from utils import *


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
        for s in self.free_symbols:
            if not s.is_nonnegative:
                raise ValueError(f'All variables used in the matrix must be declared to be non-negative.')

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

    def __str__(self):
        return str(self.matrix)

    def as_numpy(self):
        return sympy_to_numpy(self.matrix)

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

    def rref(self, indices: set[int] = None):
        return matrix_to_linalg(sub_matrix(t_matrix(self.matrix), indices)).rref()[0]
