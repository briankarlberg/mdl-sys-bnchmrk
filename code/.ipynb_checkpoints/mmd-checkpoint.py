# moda is for data modality
# ie transcrptmcs, prteomcs

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
def compute_mmd(moda_1, moda_2, gamma=1.0):
    K_XX = rbf_kernel(moda_1, moda_1, gamma=gamma)
    K_XY = rbf_kernel(moda_1, moda_2, gamma=gamma)
    K_YY = rbf_kernel(moda_2, moda_2, gamma=gamma)

    m = gexp_1.shape[0]
    n = gexp_2.shape[0]

    mmd = (np.sum(K_XX) - np.trace(K_XX)) / (m * (m - 1))
    mmd += (np.sum(K_YY) - np.trace(K_YY)) / (n * (n - 1))
    mmd -= 2 * np.sum(K_XY) / (m * n)

    # Ensure the MMD value is non-negative
    mmd = np.maximum(mmd, 0)

    return np.sqrt(mmd)