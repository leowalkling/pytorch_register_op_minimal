import os
import torch
from torch_plus import torch_plus_ops
torch.ops.load_library(os.path.join(os.path.split(__file__)[0], "myops.dll"))
myops = torch.ops.myops
