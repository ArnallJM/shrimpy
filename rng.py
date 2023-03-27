from numpy.random import default_rng
import numpy as np


DEFAULT_SEED = 0

rng = default_rng(DEFAULT_SEED)
# rng = np.random

# def reseed_rng(seed=DEFAULT_SEED):
#     global rng
#     rng = default_rng(seed)
