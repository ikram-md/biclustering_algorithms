import numpy as np

from .bicluster import Bicluster


class CCA:
    def __init__(
        self,
        sigma: float,
        alpha: float,
        missingval_indicator : int,
        nb_biclusters: int = 3,
    ) -> tuple:
        self._nb_biclusters = nb_biclusters
        self._sigma = sigma
        self._alpha = alpha
        self.biclusters = []
        self.rows_col_thr = 100
        self.missingval_indicator = missingval_indicator

    # validating the attributes
    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        if value <= 0:
            raise ValueError("Sigma value must be positive")
        self._sigma = value


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



    def handle_missing_values(self, matrix, missingval_indicator):
        """Function that handles replacing the missing values by sampling from a uniform distribution on every attribute.
        The goal is to randomize with a uniform probability of selection for every attribute.

        Args:
            matrix (_type_): Ndarray represting the matrix of values
            missingval_indicator (_type_): the indicator of the missing value (in this case it's a -1)
        """
        args = np.argwhere(matrix == missingval_indicator)

        for rind, colind in args:
            col = matrix[:, colind]

            filteredcol = col[col != missingval_indicator]

            colmin, colmax = np.min(filteredcol), np.max(filteredcol)
            matrix[rind, colind] = np.random.uniform(colmin, colmax)

    def _base_msr_score(self, submatrix, row_indices, col_indices): 
        # handling inv residues
        inv_rows = np.where(row_indices < 0)[0]
        data = submatrix[row_indices][:, col_indices]
        data_mean = np.mean(data)
        column_mean = np.mean(data, axis=0)


        if len(inv_rows) > 0:
            inv_data_rows = submatrix[np.absolute(row_indices[inv_rows])][:, col_indices]
            inv_row_mean = np.mean(inv_data_rows, axis=1)[:, np.newaxis]
            inv_residues = (-inv_data_rows + inv_row_mean - column_mean + data_mean)**2

        pos_rows = np.where(row_indices > 0)[0]
        posrows_data = submatrix[row_indices[pos_rows]][:,col_indices]
        posrows_mean = np.mean(posrows_data, axis=1)[:, np.newaxis]
        pos_residues = (posrows_data - posrows_mean - column_mean + data_mean)**2

        residues = np.concatenate((pos_residues, inv_residues), axis=0) if len(inv_rows) > 0 else pos_residues
        
        #return msr_score, row_msr, column_msr
        return np.mean(residues), np.mean(residues, axis=1) , np.mean(residues, axis=0)

    def msr_score(self, submatrix, rows, cols):
        """
        Function to calculate mean square residue score of a submatrix defined as H(I,J).
        The mean square residue score is the variance of the set of all elements in the bicluster plus the mean row variance & the mean column variance.

        Args:
            submatrix (_type_): submatrix : The bicluster matrix on which we want to calculate the mean square residue score.

        Returns:
            Float : The H(I,J) mean square residue score.
        """
        row_indices = np.nonzero(rows)[0]
        col_indices = np.nonzero(cols)[0]

        return self._base_msr_score(submatrix, row_indices, col_indices)

        
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


        _, row_msr, column_msr = self.msr_score(matrix, rows, cols)

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


    def multiple_node_deletion(self, matrix, rows, cols):
        """
            Node addition method. Defined as the second algorithm in the CC paper (2000). The goal of this algorithm is to decrease the score
        by the remaining rows and columns.

        Args:
            matrix (NArray): original data matrix.
            rows (NArray): Array of indecies that represent the rows.
            cols (NArray): Array of indecies that represent the columns.

        Returns:
            Tuple: (Rows, Columns) : Updated sigma bicluster matrix.
        """

        # verifying if row/cols threshold = 100
        print(len(rows), len(cols))
        if len(rows) < self.rows_col_thr and len(cols) < self.rows_col_thr : return 


        msr= self.msr_score(matrix, rows, cols)[0]

        converged = False
        while msr > self.sigma and not converged:

            previous_rows, previous_cols = np.copy(rows), np.copy(cols)
            msr, row_msr, column_msr= self.msr_score(matrix, rows, cols)

            # remove the rows with the highest msr
            rows_indices = np.nonzero(rows)[0]
            rows_to_remove = rows_indices[np.argwhere(row_msr > self._alpha * msr).flatten()]
            rows[rows_to_remove] = False
            
            #IMPORTANT STEP: recompute the msr after deleting the rows.
            msr, row_msr, column_msr = self.msr_score(matrix, rows, cols)


            # remove the columns with the highest msr
            bicluster_columns = np.nonzero(cols)[0]
            columns_to_remove = bicluster_columns[np.argwhere(column_msr > self._alpha * msr).flatten()]
            
            if len(columns_to_remove) > 0:
                cols[columns_to_remove] = False 
                
                msr, row_msr, column_msr = self.msr_score(matrix, rows, cols)

            # checking if nothing has been removed
            if np.array_equal(np.sort(previous_cols), np.sort(cols)) and np.array_equal(
                np.sort(previous_rows), np.sort(rows)
            ):
                converged = True



    def _col_addition_msr(self, matrix, rows, cols, msr_bij):
        """
        Function to calculate mean square residue score of a submatrix defined as H(I,J).
        The mean square residue score is the variance of the set of all elements in the bicluster plus the mean row variance & the mean column variance.
        print(rows2add)

        Args:
        submatrix (_type_): submatrix : The bicluster matrix on which we want to calculate the mean square residue score.

        Returns:
        Float : The H(I,J) mean square residue score.
        """

        row_indices, col_indices = np.nonzero(rows)[0], np.nonzero(cols)[0]
        odd_cols = np.setdiff1d(np.arange(matrix.shape[1]), col_indices)

        oddcols_data = matrix[row_indices][:,odd_cols]
        data_rows = matrix[row_indices][:, col_indices]
        row_mean = np.mean(data_rows, axis=1)

        column_mean = np.mean(oddcols_data, axis=0)
        column_rs = (oddcols_data - row_mean[:, np.newaxis] - column_mean + np.mean(data_rows)) ** 2
        
        # augment the columns based on the condition

        cols2add = np.argwhere(np.mean(column_rs, axis=0) <= msr_bij).flatten()
        if (len(cols2add) >  0):
            cols[cols2add] = True



    def _row_addition_msr(self, matrix, rows, columns, msr_bij):
        
        row_indices, col_indices = np.nonzero(rows)[0], np.nonzero(columns)[0]
        
        data = matrix[row_indices][:,col_indices]
        matrix_mean = np.mean(data)

        data_columns = matrix[:, col_indices]
        oddrows = np.setdiff1d(np.arange(matrix.shape[0]), row_indices)

        oddrows_data = matrix[oddrows][:, col_indices]
        row_mean = np.mean(oddrows_data, axis=1)[:, np.newaxis] 
        column_mean = np.mean(data_columns, axis=0)

        rows_rs = (oddrows_data - row_mean - column_mean + matrix_mean) ** 2
        rows_msr = np.mean(rows_rs, axis=1)

        # Update the set of the rows : i \in I 
        rows2add = np.argwhere(rows_msr <= msr_bij).flatten()
        rows[rows2add] = True

        # for i still not in I add the inverse rows
        row_indices = np.nonzero(rows)[0]
        
        inv_oddrows = np.setdiff1d(np.arange(matrix.shape[0]), row_indices)
        inv_row_data = matrix[inv_oddrows][:, col_indices]
        inv_row_mean=np.mean(inv_row_data, axis=1)[:, np.newaxis]
        inv_col_mean = np.mean(matrix[row_indices][:, col_indices], axis=0)

        # compute the inverse residue score
        inv_residues = (-inv_row_data +inv_row_mean -inv_col_mean +matrix_mean)**2
        inv_rows_msr =np.mean(inv_residues, axis=1)

        #extract inv rows to be added 
        invrows2add = np.argwhere(inv_rows_msr <= msr_bij).flatten()
        rows[invrows2add] = True

        #save the shape of the rows with the inverse version
        totalrows_indices = np.concatenate((rows2add, -1*invrows2add), axis=0)
        return totalrows_indices 


    def node_addition(self, data_matrix, rows, cols):
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
        totalrows, totalcols = np.nonzero(rows)[0], np.nonzero(cols)[0]
        converged = False
        while self.msr_score(data_matrix, rows, cols)[0] <= self.sigma and not converged:
            
            prev_rows,prev_cols = np.copy(rows), np.copy(cols)
            msr = self.msr_score(data_matrix, rows, cols)[0]

            # column addition 
            self._col_addition_msr(data_matrix, rows, cols, msr)
            msr = self.msr_score(data_matrix, rows, cols)[0]

            # row addition
            totalrows = self._row_addition_msr(data_matrix, rows, cols, msr)
            totalcols = np.nonzero(cols)[0]

            # check if the rows and columns are the same
            unchanged = True if np.array_equal(np.sort(rows), np.sort(prev_rows)) and np.array_equal(np.sort(cols), np.sort(prev_cols)) else False
            
            if unchanged:
                converged = True
                msr = self.msr_score(data_matrix, rows, cols)[0]
        return totalrows, totalcols
        


    def run(self, matrix):
        """Method to run the biclustering algorithm in order to generate the n number of biclusters.

        Args:
            matrix (NArray): the matrix of original data.

        Raises:
            ValueError: in case the attributes are not valid.
        """
        
        
        # clean the missing values of A by random values in range(min, max) from a normal distribution
        self.handle_missing_values(matrix, self.missingval_indicator)

        original_matrix = matrix.copy()
        
        #extract min and max values of the matrix to take samples for randomiziation
        min_val = np.min(matrix)
        max_val = np.max(matrix)

        size_rows, size_cols = original_matrix.shape
        for i in range(self._nb_biclusters):
           
            print(f'Bicluster {i+1}')
            rows, cols = np.ones(size_rows, dtype=bool), np.ones(size_cols, dtype=bool)
            self.multiple_node_deletion(
                matrix, rows, cols
            )
            self.single_node_deletion(matrix, rows, cols)

            print(f"MSR after Single + Multiple node deletion: = {self.msr_score(original_matrix, rows, cols)[0]}")
            # here we are saving the indices of the final rows and columns 
            # finalrows, finalcols = self.node_addition(original_matrix, rows, cols)
            finalrows, finalcols = np.nonzero(rows)[0], np.nonzero(cols)[0]
            # check if there aren't any new biclusters that are being discovered
            if len(finalrows) == 0 or len(finalcols) == 0:
                break
            
            bicluster = Bicluster(rows=finalrows, columns=finalcols, msr_score=self.msr_score(original_matrix, rows, cols)[0])

            # mask the discovered bicluster
            # in this step we need the mask the values but we need to pick from a uniform distribution
            bicluster_shape = (len(finalrows), len(finalcols))
            matrix[finalrows[:, np.newaxis], finalcols] = np.random.uniform(low=min_val, high=max_val, size=bicluster_shape)
            
            self.biclusters.append(bicluster)


    
    

    

    
