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
        self._msr_history = []

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

    def msr_score(self, submatrix, rows, cols):
        """
        Function to calculate mean square residue score of a submatrix defined as H(I,J).
        The mean square residue score is the variance of the set of all elements in the bicluster plus the mean row variance & the mean column variance.

        Args:
            submatrix (_type_): submatrix : The bicluster matrix on which we want to calculate the mean square residue score.

        Returns:
            Float : The H(I,J) mean square residue score.
        """

        data = submatrix[rows][:, cols]

        row_mean = np.mean(data, axis=1)[:,np.newaxis]
        column_mean = np.mean(data, axis=0)

        residues = (data - row_mean - column_mean + np.mean(data)) ** 2


        score = np.mean(residues)
        row_msr = np.mean(residues, axis=1)
        column_msr = np.mean(residues, axis=0)

        return score, row_msr, column_msr
        

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
        if msr <= self.sigma:
            return rows, cols

        column_msr_max = np.argmax(column_msr).flatten()
        row_msr_max = np.argmax(row_msr).flatten()

        if row_msr[row_msr_max] > column_msr[column_msr_max]:
            rows = np.delete(rows, row_msr_max)
        if column_msr[column_msr_max] > row_msr[row_msr_max]:
            cols = np.delete(cols, column_msr_max)
        
        print(f"Bicluster at Single node deletion score : {msr}")

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
        msr= self.msr_score(matrix, rows, cols)[0]
        changed = True
        while msr > self.sigma and changed:
            previous_rows, previous_cols = rows, cols
            
            msr, row_msr, column_msr = self.msr_score(matrix, rows, cols)

            # remove the rows with the highest msr
            rows_to_remove = np.argwhere(row_msr > self._alpha * msr).flatten()
    
            rows = np.setdiff1d(rows, rows[rows_to_remove])
            
            msr, row_msr, column_msr = self.msr_score(matrix, rows, cols)
            
            columns_to_remove = np.argwhere(column_msr > self._alpha * msr).flatten()            
            
            cols = np.setdiff1d(cols, rows[columns_to_remove])

            
            # checking if nothing has been removed
            if np.array_equal(np.sort(previous_cols), np.sort(cols)) and np.array_equal(
                np.sort(previous_rows), np.sort(rows)
            ):
                changed = False # if nothing has been removed, then just run single node deletion - this will break us out of the loop
        print(f"Bicluster at Multiple node deletion score : {msr}")
        return rows, cols  # return the rows and columns of the new bicluster

    def _node_addition_msr_columns(self, matrix, rows, columns, msr_bij):
        # find the odd columns

        odd_columns = np.setdiff1d(np.arange(matrix.shape[1]), columns)
        reduced_matrix= matrix[rows][:, odd_columns]
        reduced_row_only = matrix[rows]

        column_mean = np.mean(reduced_matrix, axis=0)
        row_mean = np.mean(reduced_row_only, axis=1)

        bij_mean = np.mean(matrix[rows][:, columns]) # single value
        residues = (reduced_matrix - row_mean[:, np.newaxis] - column_mean + bij_mean)**2

        # take the mean of the residues matrix on axis = 1 (column wise)
        column_msr = np.mean(residues, axis=0)
        cols_to_add = np.argwhere(column_msr <= msr_bij).flatten()

        return np.sort(np.concatenate((columns, cols_to_add)))

    def _node_addition_msr_row(self, matrix, rows, columns, msr_bij):
        #find the odd rows
        reduced_columns_only = matrix[rows][:, columns]
        odd_rows = np.setdiff1d(np.arange(matrix.shape[0]), rows)
        reduced_matrix = matrix[odd_rows][:, columns]

        column_mean = np.mean(reduced_columns_only, axis=0)
        row_mean = np.mean(reduced_matrix,axis=1)
        bij_mean = np.mean(matrix[rows][:, columns])

        residues = (reduced_matrix - row_mean[:,np.newaxis] - column_mean + bij_mean)**2

        # compute the row msr on axis = 0
        row_msr = np.mean(residues, axis=0)
        rows_to_add = np.argwhere(row_msr <= msr_bij).flatten()

        
        inv_row_residues = (-reduced_matrix + column_mean - row_mean[:, np.newaxis] + bij_mean)**2
        inv_row_msr = np.mean(inv_row_residues, axis=0)
        inv_rows_to_beadded = np.argwhere(inv_row_msr <= msr_bij).flatten()

        all_rowstoadd = np.concatenate((rows_to_add, inv_rows_to_beadded))

        return np.sort(np.concatenate((rows, all_rowstoadd)))
            

    # this function is supposed to decrease the score at every subroutine 
    # the msr value should always decrease
    def node_addition(self, matrix, rows, columns):
        """_summary_

        Args:
            matrix (_type_): _description_
            rows (_type_): _description_
            columns (_type_): _description_
        """
        converged = False

        while not converged:
            old_rows, old_columns = np.copy(rows), np.copy(columns)

            # compute msr current B =(I,J)
            
            msr, _,_ = self.msr_score(matrix, rows, columns)
            columns = self._node_addition_msr_columns(matrix, rows, columns, msr)
            msr, _, _ = self.msr_score(matrix, rows, columns)
            rows = self._node_addition_msr_row(matrix, rows, columns,msr)

            #This condition checks :
            # - if nothing has been added in the iterate 
            # - if threshold msr is greater than the one of this bicluster ( this is a red flag if it's true)
            if np.array_equal(np.sort(old_rows), np.sort(rows)) and np.array_equal(
                np.sort(old_columns), np.sort(columns)
            ):
                converged = True
            
        print("Bicluster score at Node addition :", msr)
        return rows, columns



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

            # Generate a random value w.r.t the normally distributed attribute
            o_matrix[r, c] = np.random.normal(
                loc=attribute_mean, scale=attribute_std
            )  

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
        

        size_rows, size_cols = original_matrix.shape
        for i in range(self._nb_biclusters):

            rows, columns = np.arange(size_rows), np.arange(size_cols)
            rows1, cols1 = self.multiple_node_deletion(
                matrix, rows, columns
            )
            rows2, cols2 = self.single_node_deletion(matrix, rows1, cols1)


            rows3, cols3 = self.node_addition(matrix, rows2, cols2)


            # Mask values from the original matrix
            
            bsc_msr, _, _ = self.msr_score(matrix, rows3, cols3)

            # append msr history 
            self._msr_history.append(bsc_msr)
            bicluster = Bicluster(rows=rows3, columns=cols3, msr_score=bsc_msr)
            
            matrix = self.randomize_values(matrix, (rows3, cols3))

            self.biclusters.append(bicluster)




    
    

    

    