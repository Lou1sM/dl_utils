from dl_utils.torch_misc import *
import numpy as np

cifarlike_dset = CifarLikeDataset(np.random.rand(100,32,32,3),np.random.randint(10,size=(100,)))
dloader = torch.utils.data.DataLoader(cifarlike_dset)
dpoint = next(iter(dloader))
print(dpoint[0].shape,dpoint[1].shape)
