from src.label_funcs import *
from pdb import set_trace

l1 = np.random.choice(10,size=10000)
l2 = np.tile(np.arange(10),1000)
translate_labellings(l1,l2,subsample_size='none')
assert unique_labels(l1) == unique_labels(l2)
assert (compress_labels(l1+1)[0] == l1).all()
assert list(label_counts(l2).values())==[1000]*10
