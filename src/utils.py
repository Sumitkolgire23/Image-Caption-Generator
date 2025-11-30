import random
import os
import pickle
from collections import Counter

import torch


def set_seed(seed=42):
    random.seed(seed)
    os.environ.setdefault('PYTHONHASHSEED', str(seed))
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
