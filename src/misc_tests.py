from dl_utils.tensor_funcs import noiseify
from dl_utils.misc import *

for i in range(100):
    for nd in [2,3,5,7]:
        assert int(n_digitify(i,nd)) == i

for inp,nd in zip(np.random.randint(10,size=6),np.random.randint(4,size=6)):
    print(f"{inp},{nd} --> {n_digitify(inp,nd)} for inp")
