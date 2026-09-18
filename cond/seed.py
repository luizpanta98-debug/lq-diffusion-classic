'''set seed for reproducibility'''
import os
import random
import numpy as np
import torch as th

def set_seed(seed=42):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)

    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)

    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

    th.use_deterministic_algorithms(True)