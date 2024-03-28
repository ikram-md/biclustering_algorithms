import csv
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import make_blobs
import numpy as np
import re
from .cca2 import CCA

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
        results = [lines[i-3:i] for i in range(3,300, 3)]

        
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
                    print(vals)
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

    def write_into_csv(self, biclusters, data_size: tuple, path: str):
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

            for _, b in enumerate(biclusters):

                writer.writerow([len(b.rows), len(
                    b.columns), b.msr_score, b.rows, b.columns])

        pd.read_csv(path, names=col_names).to_csv(
            path, encoding="utf-8", index=False)


            
    def record_msr(self, path):
        with open(path, 'r') as f:
            for index, line in enumerate(f.readlines()):
                orw, rows,columns, msr= line.split(',')[0], line.split(',')[4], line.split(',')[3], float(line.split(',')[2])
                rows = rows.replace('-', ' -')
                indices = []
                parts = rows.split()
                i = 0
                while i < len(parts):
                    if parts[i] == '-':
                        indices.append(-int(parts[i+1]))
                        i += 1
                    elif parts[i].startswith('-'):
                        indices.append(int(parts[i]))
                    
                    else:
                        indices.append(int(parts[i]))

                    i +=1
                yield msr, np.array(indices, dtype=int),np.array(columns.split(), dtype=int)
                
                    

