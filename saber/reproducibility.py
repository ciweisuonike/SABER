import random

import numpy as np
import torch


def set_global_seed(seed, deterministic=False):
    torch.use_deterministic_algorithms(bool(deterministic))
    torch.backends.cudnn.deterministic = bool(deterministic)
    if deterministic:
        torch.backends.cudnn.benchmark = False

    if seed is None:
        return

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
