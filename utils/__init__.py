from utils import utils
import random
import torch
import numpy as np


def setup_seed(seed=3047):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)的
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
