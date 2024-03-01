import numpy as np

from .bicluster import Bicluster


class CCA:
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

    def msr_score(self, submatrix, rows, cols):
        """
        Function to calculate mean square residue score of a submatrix defined as H(I,J).
        The mean square residue score is the variance of the set of all elements in the bicluster plus the mean row variance & the mean column variance.

        Args:
            submatrix (_type_): submatrix : The bicluster matrix on which we want to calculate the mean square residue score.

        Returns:
            Float : The H(I,J) mean square residue score.
        """

        data = submatrix[rows][:,cols]
        row_mean = np.mean(data, axis=1)[:, np.newaxis]
        column_mean = np.mean(data, axis=0)
        residues = (data - row_mean - column_mean + np.mean(data)) ** 2

        matrix_msr = np.mean(residues)
        row_msr = np.mean(residues, axis=1) 
        column_msr = np.mean(residues, axis=0)

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

        while msr > self._sigma:
            column_msr_max = np.argmax(column_msr)
            row_msr_max = np.argmax(row_msr)

            row_indices = np.nonzero(rows)[0]
            col_indices = np.nonzero(cols)[0]


            if row_msr[row_msr_max] >= column_msr[column_msr_max]:
                rows_to_drop = row_indices[row_msr_max]
                rows[rows_to_drop] = False

            else:
                columns_to_drop = col_indices[column_msr_max]
                cols[columns_to_drop] = False


            # recomute the msr value for the next iterate
            msr, row_msr, column_msr = self.msr_score(matrix, rows, cols)



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
        msr, row_msr, column_msr = self.msr_score(matrix, rows, cols)

        converged = True if msr <= self.sigma else False
        while not converged:

            previous_rows, previous_cols = np.copy(rows), np.copy(cols)


            # remove the rows with the highest msr
            bicluster_rows = np.nonzero(rows)[0]
            rows_to_remove = bicluster_rows[np.where(row_msr > self._alpha * msr)]
            rows[rows_to_remove] = False 
            
            # remove the columns with the highest msr
            msr, row_msr, column_msr = self.msr_score(matrix, rows, cols)
            bicluster_columns = np.nonzero(cols)[0]
            columns_to_remove = bicluster_columns[np.where(column_msr > self._alpha * msr)]
            
            if len(columns_to_remove) > 0:
                cols[columns_to_remove] = False 
            
            msr, row_msr, column_msr = self.msr_score(matrix, rows, cols)

            # checking if nothing has been removed
            if np.array_equal(previous_cols, cols) and np.array_equal(
                previous_rows, rows
            ) or msr <= self.sigma:
                converged = True

    def col_addition_msr(self, matrix, rows, columns):
        """
        Function to calculate mean square residue score of a submatrix defined as H(I,J).
        The mean square residue score is the variance of the set of all elements in the bicluster plus the mean row variance & the mean column variance.

        Args:
        submatrix (_type_): submatrix : The bicluster matrix on which we want to calculate the mean square residue score.

        Returns:
        Float : The H(I,J) mean square residue score.
        """

        data = matrix[rows][:,columns]
        data_rows = matrix[rows]

        matrix_mean = np.mean(data)
        row_mean = np.mean(data, axis=1)[:, np.newaxis] 
        column_mean = np.mean(data_rows, axis=0)
        

        column_rs = (data_rows - row_mean - column_mean + matrix_mean) ** 2
        column_msr = np.mean(column_rs, axis=0)

        return column_msr

    def row_addition_msr(self, matrix, rows, columns):
        
        data = matrix[rows][:,columns]
        data_columns = matrix[:, columns]

        matrix_mean = np.mean(data)
        row_mean = np.mean(data_columns, axis=1)[:, np.newaxis] if len(rows) > 0 else 0
        column_mean = np.mean(data, axis=0) if len(columns) > 0 else 0

        rows_rs = (data_columns - row_mean - column_mean + matrix_mean) ** 2
        rows_msr = np.mean(rows_rs, axis=1) if len(rows_rs) > 0 else 0

        # compute the inverse residue
        inverse_row_rs = (-data_columns + row_mean - column_mean + matrix_mean)**2
        inverse_row_msr = np.mean(inverse_row_rs, axis=1)
        return rows_msr, inverse_row_msr


    def node_addition(self, data_matrix, B_rows, B_cols):
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

        converged = False
        while not converged:

            prev_rows,prev_cols = np.copy(B_rows), np.copy(B_cols)
            B_msr = self.msr_score(data_matrix, B_rows, B_cols)[0]

            # column addition 
            column_msr = self.col_addition_msr(data_matrix, B_rows, B_cols)
            column_to_add = np.where(column_msr <= B_msr)[0]
            B_cols[column_to_add] = True

            # row addition
            B_msr = self.msr_score(data_matrix, B_rows, B_cols)[0]
            row_msr, inverse_row_msr = self.row_addition_msr(data_matrix, B_rows, B_cols)
            rows_to_add = np.where(np.logical_or(row_msr <= B_msr, inverse_row_msr <= B_msr))[0]
            B_rows[rows_to_add] = True

            # checl if the rows and columns are the same
            if np.array_equal(B_rows, prev_rows) and np.array_equal(B_cols, prev_cols):
                converged = True


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

        #extract min and max values of the matrix to take samples for randomiziation
        min_val = np.min(matrix)
        max_val = np.max(matrix)

        size_rows, size_cols = original_matrix.shape
        for i in range(self._nb_biclusters):
            rows, cols = np.ones(size_rows, dtype=bool), np.ones(size_cols, dtype=bool)
        

            self.multiple_node_deletion(
                matrix, rows, cols
            )
            self.single_node_deletion(matrix, rows, cols)
            self.node_addition(matrix, rows, cols)
            

            

            # extract the indecies of the rows and columns of the bicluster
            bicluster_rows, bicluster_cols, bicluster_msr= np.nonzero(rows)[0], np.nonzero(cols)[0], self.msr_score(matrix, rows, cols)[0]
            
            # check if there aren't any new biclusters that are being discovered
            if len(bicluster_rows) == 0 or len(bicluster_cols) == 0:
                break
            
            # mask the discovered bicluster
            if i < self._nb_biclusters: 
                bicluster_shape = (len(bicluster_rows), len(bicluster_cols))
                
                matrix[bicluster_rows[:, np.newaxis], bicluster_cols] = np.random.uniform(low=min_val, high=max_val, size=bicluster_shape)

            bicluster = Bicluster(rows=bicluster_rows, columns=bicluster_cols, msr_score=bicluster_msr)
            
            self.biclusters.append(bicluster)


    
    

    

    
