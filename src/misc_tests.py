from dl_utils.tensor_funcs import noiseify
from dl_utils.misc import *

for i in range(100):
    for nd in [2,3,5,7]:
        assert int(n_digitify(i,nd)) == i

for inp,nd in zip(np.random.randint(10,size=30),np.random.randint(1,4,size=30)):
    print(f"{inp},{nd} --> {n_digitify(inp,nd)} for inp")

cifarlike_dset = CifarLikeDataset(np.random.rand(100,32,32,3),np.random.randint(10,size=(100,)))
dloader = torch.utils.data.DataLoader(cifarlike_dset)
dpoint = next(iter(dloader))
print(dpoint[0].shape,dpoint[1].shape)
