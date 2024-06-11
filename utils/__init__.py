from utils import utils
import random
import torch
import numpy as np


def setup_seed(seed=3047):
    torch.manual_seed(seed)  # 为cpu生成种子，确保每次运行.py文件时，生成的随机数都是固定的
    torch.cuda.manual_seed_all(seed)  # 为所有的 GPU 设置种子用于生成随机数，以使得结果是确定的
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # 每次返回的卷积算法将是确定的，配合上设置Torch的随机种子为固定值，保证每次运行网络的时候相同输入的输出是固定的
