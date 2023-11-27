import numpy as np
import pandas as pd


class Biclustering:
    def __init__(
        self,
        sigma: int,
        alpha: int,
        iterations: int = 100,
    ) -> tuple:
        self.iterations = iterations
        self.sigma = sigma
        self.alpha = alpha

    def mean_sqaure_residue(
        self, start: tuple, submatrix, submatrix_mean: float
    ) -> float:
        I, J = submatrix.shape[0], submatrix.shape[1]
        _i, _j = start
        element = submatrix[_i, _j]
        row_mean = (1 / J) * np.sum(submatrix[_i, :])
        column_mean = (1 / I) * np.sum(submatrix[:, _j])
        return (element - row_mean - column_mean + submatrix_mean) ** 2

    def mean_square_residue_score(self, submatrix):
        """ 
        Function to calculate mean square residue score of a submatrix defined as H(I,J).
        The mean square residue score is the variance of the set of all elements in the bicluster plus the mean row variance & the mean column variance. 

        Args:
            submatrix (_type_): submatrix : The bicluster matrix on which we want to calculate the mean square residue score.

        Returns:
            Float : The H(I,J) mean square residue score 
        """
        n, m = submatrix.shape[0], submatrix.shape[1]
        submatrix_mean = np.mean(submatrix)
        msr_scores = np.zeros((n * m))
        for i in range(n):
            for j in range(m):
                msr = self.mean_sqaure_residue((i, j), submatrix, submatrix_mean)
                msr_scores[i + j] = msr
        return (1 / (n * m)) * np.sum(msr_scores)

    def single_node_deletion(
        self,
        submatrix,
    ):
        """
        Single node deletion algorithm from algorithms 1 of the first paper.
        This algorithm removes a single node from the general expressions data matrix by computing and
        evaluating the mean square residue of the submatrix per rows and columns.
        Then eventually taking the largest score and removing the equivalent row or column. The aim of this algorithm is reduce the score of the bicluster A(IJ)

        Args:
            submatrix (submatrix): a copy of the general expressions data matrix on which the clustering is performed.


        Returns:
            updated_submatrix: the resulted sigma bicluster matrix.
        """
        submatrix_mean = np.mean(submatrix)

        d_i = np.zeros(submatrix.shape[1])
        d_j = np.zeros(submatrix.shape[0])

        msrs = self.mean_square_residue_score(submatrix)
        is_valid = lambda x: x > self.sigma

        if not is_valid(msrs):
            return submatrix.copy()

        # compute all the score for rows
        for i in range(submatrix.shape[0]):
            for j in range(submatrix.shape[1]):
                d_i[j] = (1 / submatrix.shape[1]) * self.mean_sqaure_residue(
                    (i, j), submatrix, submatrix_mean
                )

        # compute the score for all columns
        for j in range(submatrix.shape[1]):
            for i in range(submatrix.shape[0]):
                d_j[i] = (1 / submatrix.shape[0]) * self.mean_sqaure_residue(
                    (i, j), submatrix, submatrix_mean
                )

        # compare the larger score between the rows and the columns
        updated_submatrix = submatrix.copy()
        max_di, max_dj = np.argmax(d_i), np.argmax(d_j)
        if d_i[max_di] > d_j[max_dj]:
            print(f"Removing the {max_di}th row")
            updated_submatrix = np.delete(submatrix, max_di, axis=0)
        else:
            print(f"Removing the {max_dj}th column")
            updated_submatrix = np.delete(submatrix, max_di, axis=0)

        # create a new matrix with the removed rows & columns
        return updated_submatrix
    
    def multiple_node_deletion(self, submatrix):
