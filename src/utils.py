import random
import numpy as np

def set_all_seeds(seed: int = 7):
    random.seed(seed)
    np.random.seed(seed)
