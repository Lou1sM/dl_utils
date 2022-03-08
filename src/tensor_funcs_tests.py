from dl_utils.tensor_funcs import *
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
loaded_arr = np_load_all('/home/louis/feature_viz/dataset/alexnet/X',restrict=25)
print(loaded_arr.shape)
loaded_arr = np_load_all('/home/louis/feature_viz/dataset/alexnet/X')
print(loaded_arr.shape)
print("Should see three images displayed followed by a shape error")
display_image(torch.rand(3,28,28).cuda())
display_image(torch.rand(1,28,28).cuda())
display_image(torch.rand(224,224,1).cuda())
display_image(torch.rand(224,224,4).cuda())
