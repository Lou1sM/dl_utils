from dl_utils import label_funcs
from pdb import set_trace
import numpy as np

print("\nTESTING LABEL_FUNCS\n")

l1 = np.random.choice(10,size=10000)
l2 = np.tile(np.arange(10),1000)
label_funcs.translate_labellings(l1,l2,subsample_size='none')
print("Should see a user warning:")
label_funcs.translate_labellings(np.array([]),np.array([]),subsample_size='none')
assert label_funcs.unique_labels(l1) == label_funcs.unique_labels(l2)
assert (label_funcs.compress_labels(l1+1)[0] == l1).all()
assert list(label_funcs.label_counts(l2).values())==[1000]*10
print(f"Should see a number close to 0.1:\n\t{label_funcs.accuracy(l1,l2)}")
print(f"Should see a number close to 0.1:\n\t{label_funcs.mean_f1(l1,l2)}")

babled = np.random.randint(10,size=(5,1000))
babled1 = [np.random.randint(10,size=(1000)) for _ in range(5)]
babled_wrong = np.random.randint(10,size=(5,1000,6))
label_funcs.debable(babled,pivot='none')
label_funcs.debable(babled1,pivot='none')
label_funcs.dummy_labels(7,110)
print("Should see TranslationError:")
label_funcs.debable(babled_wrong,pivot='none')
