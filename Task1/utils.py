import os
import torch
import torch.nn as nn

from torch.nn import functional as F

def checkFolder(dir):
    if not os.path.exists(dir):
        umask = os.umask(0)
        try:
            os.makedirs(dir, 0o755, exist_ok = True)
        finally:
            os.umask(umask)