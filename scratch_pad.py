import numpy as np
import pandas as pd
from implementation.biclustering_ import Biclustering
import time

from implementation.data_factory import DataFactory


#     I have reviewed the 3 files changed
#     I could reproduce the results for finding 5, 20, and 30 biclusters
#     This is a very nice implementation of the CC algorithm. Congrats !


raw_data = pd.read_csv('./data/yeast_matrix.csv')

# extract & convert all the columns
data = pd.DataFrame(raw_data)

# converting the types acoordingly
df =data.astype('float')
# extracting a test set
# timing the execution for 5 biclusters 

test_set = data.astype('float').to_numpy()


start_time = time.time()
exp1 = Biclustering(sigma=300, alpha=1.2, nb_biclusters=100)
exp1.run(test_set)


# evaluating the results
df_fact = DataFactory()
df_fact.write_into_csv(1, exp1.biclusters,test_set.shape)
df_fact.measure_error("./data/biclusters_yeast_data.csv", "./experiments/exp-1.csv")