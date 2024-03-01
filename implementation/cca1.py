import numpy as np

from .bicluster import Bicluster


class Biclustering:
    def __init__(
        self,
        sigma: float,
        alpha: float,
        nb_biclusters: int = 3,
    ) -> tuple:
        self._nb_biclusters = nb_biclusters
        self._sigma = sigma
        self._alpha = alpha
        self.biclusters = []
        self.rows_col_thr = 100

    # validating the attributes
    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        if value <= 0:
            raise ValueError("Sigma value must be positive")
        self._sigma = value

    @property
    def alpha(self):
        return self.alpha

    @alpha.setter
    def alpha(self, alpha):
        if alpha <= 0:
            raise ValueError("Alpha value must be strictly positive")
        self._alpha = alpha

    @property
    def nb_biclusters(
        self,
    ):
        return self._nb_biclusters

    @nb_biclusters.setter
    def nb_biclusters(self, nb_biclusters):
        if nb_biclusters > 50:
            raise ValueError("Number of biclusters must not be greater than 50")
        self._nb_biclusters = nb_biclusters

    def msr_score(self, submatrix, rows, cols, inverse_msr_score=False):
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

        row_mean = np.mean(data, axis=1)[:, np.newaxis] if len(rows) > 0 else 0

        column_mean = np.mean(data, axis=0) if len(cols) > 0 else 0

        residues = (data - row_mean - column_mean + np.mean(data)) ** 2

        if inverse_msr_score:
            residues = (-data + row_mean - column_mean + np.mean(data)) ** 2

        matrix_msr = np.mean(residues)

        row_msr = np.mean(residues, axis=1) if len(residues) > 0 else 0

        column_msr = np.mean(residues, axis=0) if len(residues) > 0 else 0

        return matrix_msr, row_msr, column_msr

    def single_node_deletion(self, matrix, rows, cols):
        """Single node deletion algorithm. Defined as the second algorithm in the CC paper (2000). The goal of this algorithm
        is to delete the row I or the column J with the highest mean square residue score.


        Args:
            matrix (NArray): the original data matrix.
            rows (NArray): the set of original rows (first bicluster)
            cols (NArray): the set of original columns (first bicluster)

        Returns:
            Tuple of rows & columns : Updated sigma bicluster matrix
        """

        msr, row_msr, column_msr = self.msr_score(matrix, rows, cols)
        
        if msr > self._sigma:
            column_msr_max = np.argmax(column_msr).flatten()

            row_msr_max = np.argmax(row_msr).flatten()

            if row_msr[row_msr_max] > column_msr[column_msr_max]:

                rows = np.delete(rows, row_msr_max)
            if column_msr[column_msr_max] > row_msr[row_msr_max]:

                cols = np.delete(cols, column_msr_max)
        # extract the rows and columns of the first bicluster
        return rows, cols

    def multiple_node_deletion(self, matrix, rows, cols):
        """Node addition method. Defined as the second algorithm in the CC paper (2000). The goal of this algorithm is to decrease the score
        by the remaining rows and columns.

        Args:
            matrix (NArray): original data matrix.
            rows (NArray): Array of indecies that represent the rows.
            cols (NArray): Array of indecies that represent the columns.

        Returns:
            Tuple: (Rows, Columns) : Updated sigma bicluster matrix.
        """

        converged = False
        while not converged:
            previous_rows, previous_cols = rows, cols
            msr, row_msr, column_msr = self.msr_score(matrix, rows, cols)


            # remove the rows with the highest msr
            rows_to_remove = np.argwhere(row_msr > self._alpha * msr).flatten()

            rows = (
                np.delete(rows, rows_to_remove)
                if len(rows_to_remove) > 0
                else rows
            )

            msr, row_msr, column_msr = self.msr_score(matrix, rows, cols)
            
            columns_to_remove = np.argwhere(column_msr > self._alpha * msr).flatten()
                        
            cols = (
                np.delete(cols, columns_to_remove)
                if len(columns_to_remove) > 0
                else cols
            )
            # checking if nothing has been removed

            if np.array_equal(previous_cols, cols) and np.array_equal(
                previous_rows, rows
            ):
                converged = True
                # if nothing has been removed, then just run single node deletion

        return rows, cols  # return the rows and columns of the new bicluster

    def node_addition(self, A_matrix, B_rows, B_cols):
        """Node addition method which is described as Algorithm n°3 in the CC paper. Page 6. The goal of this step is to maximize the bicluster size
        with respect to the total bicluster score

        Args:
            A_matrix (NDArray): original data matrix.
            B_rows (_type_): NDArray representing the rows of the bicluster.
            B_cols (_type_): NDArray representing the columns of the

        Returns:
            _type_: Either updated bicluster rows & columns, otherwise if nothing is added. The original rows, columns are returned.
        """
        # copying the orginal size of the bicluster for convergence later.

        # compute all the row, column means, matrix mean and H(b) of matrix B

        i = 0
        converged = False
        while not converged and i < 10:
            prev_B_rows, prev_B_cols = B_rows, B_cols

            msr_b = self.msr_score(A_matrix, B_rows, B_cols)[0]

            # extract the odd columns
            odd_columns = np.setdiff1d(np.arange(A_matrix.shape[1]), B_cols)
            # compute msr for every column  and compare it to the msr of the bicluster in itself
            odd_columns_msr = self.msr_score(A_matrix, B_rows, odd_columns)[2]

            odd_columns_to_add = np.argwhere(odd_columns_msr <= msr_b).flatten()

            # recompute msr with concatenation of the new columns
            if len(odd_columns_to_add) > 0:
                B_cols = np.concatenate(
                    (B_cols, odd_columns[odd_columns_to_add]), axis=None
                )  # concatenate the new columns with the previous ones as an array

            # else do the same thing for rows
            msr_b = self.msr_score(A_matrix, B_rows, B_cols)[0]

            odd_rows = np.setdiff1d(np.arange(A_matrix.shape[0]), B_rows)

            # compute msr for every column  and compare it to the msr of the bicluster in itself
            odd_rows_msr = self.msr_score(A_matrix, odd_rows, B_cols)[1]

            odd_rows_to_add = np.argwhere(odd_rows_msr <= msr_b).flatten()
            # update the bicluster rows
            if len(odd_rows_to_add) > 0:
                B_rows = np.concatenate((B_rows, odd_rows[odd_rows_to_add]), axis=None)

            # find rows that are still not added - compute their inverse msr score
            inverse_rows = np.setdiff1d(np.arange(A_matrix.shape[0]), B_rows)

            inverse_rows_msr = self.msr_score(A_matrix, inverse_rows, B_cols)[1]

            inverse_rows_to_add = np.argwhere(inverse_rows_msr <= msr_b).flatten()

            if len(inverse_rows_to_add):
                B_rows = np.concatenate(
                    (B_rows, inverse_rows[inverse_rows_to_add]), axis=None
                )

            # compare if we made progress in this iterate
            if len(B_rows) == len(prev_B_rows) and len(B_cols) == len(prev_B_cols):
                converged = True
            else:
                prev_B_rows, prev_B_cols = B_rows, B_cols

        return B_rows, B_cols

    # the final algorithm number 4

    def randomize_values(self, o_matrix, brc: tuple):
        """Replaces the elements of the bicluster with random values in the original matrix A'.
        The random values are drawn from a normal distribution with mean : mean of the attribute (column in bicluster D) and std : std of the same attribute (column).


        Args:
            o_matrix (NDArray): Original matrix A' (as denoted in step D in the cc algorithm).
            brc (tuple): A tuple (NDArray,NDArray) representing the rows & columns of bicluster D.
        Returns:
            A reset of the original matrix A'  (please refer to step 4 in Algorithm 4 page = 7 of the CC paper.)
        """

        # NOTE: For this example we will suppose that the matrix is already normalized
        d_bicluster = o_matrix[brc[0], :]
        d_bicluster = d_bicluster[
            :, brc[1]
        ].flatten()  # extract the elements of the bicluster matrix into a numpy array

        # Find the indicies of the matching elements in the original matrix.
        indecies = [np.argwhere(o_matrix == el)[0] for el in d_bicluster]
        for index in indecies:
            # Find the column corresponding to the matching element

            r, c = index

            attribute = o_matrix[:, c]  # extract the column from the orginal matrix

            attribute_mean = np.mean(attribute)

            attribute_std = np.std(attribute)

            randomized_val = np.random.normal(
                loc=attribute_mean, scale=attribute_std
            )  # Generate a random value w.r.t the normally distributed attribute
            o_matrix[r, c] = randomized_val

        return o_matrix

    def run(self, matrix):
        """Method to run the biclustering algorithm in order to generate the n number of biclusters.

        Args:
            matrix (NArray): the matrix of original data.

        Raises:
            ValueError: in case the attributes are not valid.
        """

        # clean the missing values of A by random values in range(min, max) from a normal distribution
        original_matrix = matrix.copy()
        
        min_value = np.min(original_matrix)
        max_value = np.max(original_matrix)

        size_rows, size_cols = original_matrix.shape
        for i in range(self._nb_biclusters):

            original_rows, original_cols = np.arange(size_rows), np.arange(size_cols)
            B_rows, B_cols = self.multiple_node_deletion(
                matrix, original_rows, original_cols
            )
            C_rows, C_cols = self.single_node_deletion(matrix, B_rows, B_cols)


            D_rows, D_cols = self.node_addition(matrix, C_rows, C_cols)
            
            
            # mask the values from the original matrix
            # bicluster_shape = (len(D_rows), len(D_cols))
            # matrix[D_rows[:,np.newaxis],D_cols] = np.random.uniform(low=min_value, high=max_value, size=bicluster_shape)

            bsc_msr = self.msr_score(matrix, D_rows, D_cols)
            bicluster = Bicluster(rows=D_rows, columns=D_cols, msr_score=bsc_msr[0])
            
            self.biclusters.append(bicluster)

            # randomize the values in rows and columns D in A
            matrix = self.randomize_values(matrix, (D_rows, D_cols))


    
    

    

    
