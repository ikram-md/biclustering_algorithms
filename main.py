import numpy as np
import pandas as pd
import os
import time
import pandas as pd

# from ccp.data_factory import DataFactory
# from ccp.cca2 import CCA
from sklearn.preprocessing import OneHotEncoder







# # Experiment parameters
# SIGMA = 300 #biclsutX = [['Jeans', 1, 5], ['Shirt', 3, 4], ['Cap', 1, 4],['Apple', 0, 1]] # the array of values are all sorted.


enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X)
print(enc.categories_)
print(enc.transform([['Apple', 2, 1]],).toarray())


# Yeast matrix
data = pd.read_csv(f'{os.getcwd()}/data/bic_dataset_test/hetdata50_nbins5_01_data.tsv', delimiter='\t')
print(data.value_counts())
# ALPHA =1.2 #node addition threshold
# WRITE_FILE_PATH = f"{os.getcwd()}/data/yeast_expression.csv"
# FEATURE_SIZE = 17 # fixed value for the yeast data matrix
# NB_BICLUSTERS = 100


# cca = CCA(SIGMA, ALPHA, nb_biclusters=NB_BICLUSTERS,missingval_indicator=-1)
# data_fact = DataFactory()

# columns = np.array([f"Cond{i+1}" for i in range(FEATURE_SIZE)])
# df = pd.read_csv(WRITE_FILE_PATH, names=columns)
# column_names = df.columns.to_list() #extract the name of
# data = df.to_numpy()


# EXP_INDEX = 5
# cca.run(data) 
# data_fact.write_into_csv(cca.biclusters, data.shape, f"{os.getcwd()}/experiments/yeast-matrix/cca2/exp-{EXP_INDEX}.csv")


