from src.tensor_funcs import *
import torch


t = torch.randn((10,3,2))
to_float_tensor(t)
add_colour_dimension(t)
noiseify(t,0.1)
t1 = torch.randn((3,2))
oheify(t1)
numpyify(t)
t2 = torch.randn((1))
print_tensors(t2,t2,t2)
