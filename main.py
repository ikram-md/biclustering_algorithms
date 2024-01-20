import numpy as np
from sklearn.datasets import make_blobs
from implementation.biclustering_ import Biclustering


# Testing the algorithm with guassian clusters
toy_data_set, y_toy = make_blobs(n_samples=20, centers=3, n_features=10)


toy_rows, toy_cols = toy_data_set.shape

biclustering_ = Biclustering(sigma=0.3, alpha=0.5, nb_biclusters=3)

print(
    "Score of the original matrix : ",
    biclustering_.msr_score(
        toy_data_set,
        np.arange(toy_rows),
        np.arange(toy_cols),
    )[0],
)


biclustering_.run(toy_data_set)
for b in biclustering_.biclusters:
    print(b.rows)
    print(b.columns)
    print(b.msr_score)
