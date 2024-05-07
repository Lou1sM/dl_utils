from dl_utils.torch_misc import *
import numpy as np


def test_dset(dset):
    dloader = torch.utils.data.DataLoader(dset)
    dpoint = next(iter(dloader))
    if isinstance(dpoint,tuple) or isinstance(dpoint,list):
        print(dpoint[0].shape,dpoint[1].shape)
    else:
        assert isinstance(dpoint,torch.Tensor)
        print(dpoint[0].shape)

cifarlike_dset = CifarLikeDataset(np.random.rand(100,32,32,3),np.random.randint(10,size=(100,)))
test_dset(cifarlike_dset)

np_dset = NPDataset(np.random.rand(100,32,32,3),np.random.randint(10,size=(100,)))
test_dset(np_dset)

np_dset = NPDataset(np.random.rand(100,32,32,3))
test_dset(np_dset)
