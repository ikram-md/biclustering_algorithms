import numpy as np
import pandas as pd
import time
from implementation.cca1 import Biclustering
from implementation.cca2 import CCA

from implementation.data_factory import DataFactory


#     I have reviewed the 3 files changed
#     I could reproduce the results for finding 5, 20, and 30 biclusters
#     This is a very nice implementation of the CC algorithm. Congrats !
raw_data = pd.read_csv('./data/yeast_matrix.csv')

# extract & convert all the columns
data = pd.DataFrame(raw_data)

# converting the types acoordingly
df =data.astype('float')

def cc_experiment(results_path : str, exp_index : int, implementation):
    test_set = data.astype('float').to_numpy()

    # running an experiment with the CCA1
    start_time = time.time()
    exp1 = implementation(sigma=300, alpha=1.2, nb_biclusters=100)
    exp1.run(test_set)
    exp_index = 11


    # evaluating the results
    df_fact = DataFactory()
    df_fact.write_into_csv(exp_index, exp1.biclusters,test_set.shape, f"{results_path}/exp-{exp_index}.csv")


cc_experiment('./experiments/cca1', 13, Biclustering)
# cc_experiment('./experiments/cca2', 12, CCA)