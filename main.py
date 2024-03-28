import numpy as np
import pandas as pd
import os
import time
import pandas as pd

from ccp.data_factory import DataFactory
from ccp.cca2 import CCA


# Yeast matrix


# Experiment parameters
SIGMA = 300 #biclsuter threshold
ALPHA =1.2 #node addition threshold
WRITE_FILE_PATH = f"{os.getcwd()}/data/yeast_expression.csv"
FEATURE_SIZE = 17 # fixed value for the yeast data matrix
NB_BICLUSTERS = 100


cca = CCA(SIGMA, ALPHA, nb_biclusters=NB_BICLUSTERS,missingval_indicator=-1)
data_fact = DataFactory()

columns = np.array([f"Cond{i+1}" for i in range(FEATURE_SIZE)])
df = pd.read_csv(WRITE_FILE_PATH, names=columns)
column_names = df.columns.to_list() #extract the name of
data = df.to_numpy()


EXP_INDEX = 5
cca.run(data) 
data_fact.write_into_csv(cca.biclusters, data.shape, f"{os.getcwd()}/experiments/yeast-matrix/cca2/exp-{EXP_INDEX}.csv")


