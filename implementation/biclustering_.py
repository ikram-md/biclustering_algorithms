import numpy as np
import pandas as pd


class Biclustering:
    def __init__(
        self,
        sigma: float,
        alpha: float,
        nb_biclusters: int = 3,
    ) -> tuple:
        self.nb_biclusters = nb_biclusters
        self.sigma = sigma
        self.alpha = alpha
        self.biclusters = []

    def msr_score(self, submatrix, rows, cols):
        """
        Function to calculate mean square residue score of a submatrix defined as H(I,J).
        The mean square residue score is the variance of the set of all elements in the bicluster plus the mean row variance & the mean column variance.

        Args:
            submatrix (_type_): submatrix : The bicluster matrix on which we want to calculate the mean square residue score.

        Returns:
            Float : The H(I,J) mean square residue score.
        """
        data = submatrix[rows]
        data = data[:, cols]

        row_mean = np.mean(data, axis=1)

        column_mean = np.mean(data, axis=0)
        residues = (data - row_mean[:, np.newaxis] - column_mean + np.mean(data)) ** 2

        matrix_msr = np.mean(residues)
        row_msr = np.mean(residues, axis=1)
        column_msr = np.mean(residues, axis=0)
        return matrix_msr, row_msr, column_msr

    def single_node_deletion(self, matrix, rows, cols):
        nb_rows, nb_columns = matrix.shape
        t_rows = []
        t_columns = []
        msr, row_msr, column_msr = self.msr_score(matrix, rows, cols)
        if msr > self.sigma:
            column_msr_max = np.argmax(column_msr)
            row_msr_max = np.argmax(row_msr)

            if row_msr[row_msr_max] > column_msr[column_msr_max]:
                t_rows = [i for i in range(nb_rows) if i != row_msr_max]
                t_columns = list(range(nb_columns))

            if column_msr[column_msr_max] > row_msr[row_msr_max]:
                t_rows = list(range(nb_rows))
                t_columns = [i for i in range(nb_columns) if i != column_msr_max]

        # extract the rows and columns of the first bicluster
        return np.array(t_rows), np.array(t_columns)

    def msr_col_addition(self, matrix, rows, cols):
        odd_cols = np.array([i for i in range(matrix.shape[1]) if i not in cols])
        _, _, column_msr = self.msr_score(matrix, rows, odd_cols)
        return column_msr

    def msr_row_addition(self, matrix, rows, cols):
        odd_rows = np.array([i for i in range(matrix.shape[0]) if i not in rows])
        _, row_msr, _ = self.msr_score(matrix, odd_rows, cols)
        data = matrix[:, cols]
        data = data[rows]

        row_mean = np.mean(data, axis=1)
        column_mean = np.mean(data, axis=0)
        inverse_row_residues = (
            -data + row_mean[:, np.newaxis] - column_mean + np.mean(data)
        ) ** 2
        inverse_row_msr = np.mean(inverse_row_residues, axis=1)
        return row_msr, inverse_row_msr

    def multiple_node_deletion(self, matrix, rows, cols):
        data = matrix[rows]
        data = data[:, cols]
        nb_rows, nb_columns = data.shape
        msr, row_msr, column_msr = self.msr_score(matrix, rows, cols)
        previous_rows, previous_cols = np.array(rows), np.array(cols)
        # evaluating the general score for the matrix

        converged = False
        while not converged:
            # remove the rows with the highest msr
            rows_to_remove = np.argwhere(row_msr > self.alpha * msr)[:, 0]
            rows = np.array([i for i in range(nb_rows) if i not in rows_to_remove])
            msr, row_msr, column_msr = self.msr_score(matrix, rows, cols)

            if msr > self.sigma:
                columns_to_remove = np.argwhere(column_msr > self.alpha * msr)[:, 0]
                cols = np.array(
                    [i for i in range(nb_columns) if i not in columns_to_remove]
                )
            if (
                previous_rows.shape[0] != rows.shape[0]
                and previous_cols.shape[0] != cols.shape[0]
            ):
                converged = True

        return np.array(rows), np.array(
            cols
        )  # return the rows and columns of the new bicluster

    def node_addition(self, matrix, rows, cols):
        """Node addition of rows and columns (defined as algorithm n°3 in the CC paper)

        Args:
            matrix (NPArray): _description_
            rows (NPArray): _description_
            cols (NPArray): _description_

        Returns:
            _type_: _description_
        """
        previous_rows, previous_cols = np.array(rows), np.array(cols)

        is_identical = lambda x, y: x.shape[0] == y.shape[0]

        # computing the score of sigma bicluster
        converged = False

        while not converged:
            _, _, msr = self.msr_score(matrix, rows, cols)

            odd_cols_msr = self.msr_col_addition(matrix, rows, cols)

            cols_to_add = np.argwhere(odd_cols_msr <= msr)[:, 0]
            cols = np.concatenate([cols, cols_to_add])

            _, _, msr = self.msr_score(matrix, rows, cols)

            row_msr, inverse_row_msr = self.msr_row_addition(matrix, rows, cols)
            rows_to_add = np.argwhere(
                np.logical_or(row_msr <= msr * msr, inverse_row_msr <= msr)
            )[:, 0]
            rows = np.concatenate([rows, rows_to_add])

            if is_identical(previous_rows, rows) and is_identical(previous_cols, cols):
                converged = True
        return rows, cols

    # the final algorithm number 4
