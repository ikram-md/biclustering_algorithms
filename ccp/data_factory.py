import csv
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import make_blobs
import numpy as np
import re


class DataFactory:
    def _preprocess_data(self, file_path: str):
        """_summary_

        Args:
            file_path (str): _description_

        Yields:
            _type_: _description_
        """
        file = open(file=file_path)
        topology_lines = file.readlines()
        for line in topology_lines:
            cleaned_line = re.split(r"\s+|\s(?=[^-0-9])", line)
            cleaned_line = [s for s in cleaned_line if s]
            s = " ".join(cleaned_line)
            numbers_str = s.replace('-', ' -').split()
            numbers = np.array([int(num_str) for num_str in numbers_str])
            yield numbers

    
    def _preprocess_results(self, filepath : str, writefilepath : str, n=1):
        """_summary_

        Args:
            filepath (str): _description_
            writefilepath (str): _description_
            n (int, optional): _description_. Defaults to 1.
        """
        # post on stack overflow
        lines = open(filepath, 'r').readlines()
        results = []
        i = 1
        while i <= len(lines):
            if i % 3 == 0:
                results.append(lines[i-3:i+3])
            i += 1
        
        # clean and write the data into a csv file
        with open(writefilepath, "a") as f:
            writer = csv.writer(f)
            for row in results:
                vals = []
                # clean the rows and write them into csv - a row is composed of 3 strings
                for index, s in enumerate(row):
                    if(index == 0):
                        validstring = s.strip().split()
                        [vals.append(num) for num in validstring]
                    else:
                        vals.append(s.strip())
                writer.writerow(vals)



        

    def _consume_preprocess_data(self, read_file: str, write_file: str):
        """_summary_

        Args:
            read_file (str): _description_
            write_file (str): _description_
        """
        # Function that write the results into a file
        with open(write_file, "a") as f:
            writer = csv.writer(f)
            for row in self._preprocess_data(read_file):
                writer.writerow(row)

    def clean_file(self, read_file: str, write_file: str):
        """A wrapper function for the clean function

        Args:
            read_file (str): _description_
            write_file (str): _description_
        """
        self._consume_preprocess_data(read_file, write_file, )

    def bicluster_coverage(self, rows, columns, original_size: tuple):
        """Function that computes the coverage of biclustered genes & conditions from the total.
        Args:
            rows (_type_): _description_
            columns (_type_): _description_
            original_size (tuple): _description_

        Returns:
            _type_: _description_
        """

        # TODO: add the total coverage on the entire matrix
        row_coverage = (len(set(rows)) * 100)/original_size[0]

        columns_coverage = (len(set(columns)) * 100)/original_size[1]

        return row_coverage, columns_coverage

    def write_into_csv(self, exp_index: int, biclusters, data_size: tuple, path: str):
        """Method to write the results into a csv file for further processing by other algorithms. Does not return anything.

        Args:
            exp_index (int): index of the experiment.
            biclusters (NDArray): list of generated biclusters.
            data_size (tuple): shape of the original bicluster matrix.
        """
        col_names = ["rows", "columns", "msr",
                     "row_coverage", "column_coverage"]

        with open(path, "a") as f:
            writer = csv.writer(f)

            for _, bicluster in enumerate(biclusters):

                cov_rows, cov_columns = self.bicluster_coverage(
                    bicluster.rows, bicluster.columns, data_size)

                writer.writerow([len(bicluster.rows), len(
                    bicluster.columns), bicluster.msr_score, cov_rows, cov_columns])

        pd.read_csv(path, names=col_names).to_csv(
            path, encoding="utf-8", index=False)

    def measure_error(self, path1, path2):
        """Method to measure the error between the original data and the predicted one. This method only measures the MSE score for the MSR of the generated biclusters

        Args:
            path1 (str): path of the file with true MSR values.
            path2 (str): path of the file with predicted MSR values.
        Returns:
            (float): MSE score.
        """
        # TODO: process the rows and the columns

        # preprocessing and parsing the data
        df1 = pd.read_csv(
            path1, names=['rows', "columns", 'msr', 'index_rows', 'index_cols'])
        df2 = pd.read_csv(path2)

        # sort by msr score in order to compute the error
        df1_sorted = df1.sort_values(by='msr', ascending=False)
        df2_sorted = df2.sort_values(by='msr', ascending=False)

        # extracting the data from csv files
        total_rows1, total_cols1, msr1 = df1_sorted['rows'].to_numpy(
            dtype=int), df1_sorted['columns'].to_numpy(dtype=int), df1_sorted['msr'].to_numpy(dtype=float)
        total_rows2, total_cols2, msr2 = df2_sorted['rows'].to_numpy(
            dtype=int), df2_sorted['columns'].to_numpy(dtype=int), df2_sorted['msr'].to_numpy(dtype=float)

        # return msr error rate
        return mean_absolute_error(msr1, msr2)
