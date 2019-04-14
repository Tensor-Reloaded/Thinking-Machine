import random
import numpy as np
import torch


def reset_seed(seed):
    """
    Sets seed of all random number generators used to the same seed, given as argument
    WARNING: for full reproducibility of training, torch.backends.cudnn.deterministic = True is also needed!
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
