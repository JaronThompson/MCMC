import numpy as np
import sklearn.datasets

X, y = sklearn.datasets.make_regression(n_samples=100, n_features=15,
    n_informative=10, n_targets=1, noise=5)

np.savetxt('features.csv', X)
np.savetxt('targets.csv', y)
