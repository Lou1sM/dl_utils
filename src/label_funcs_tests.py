from dl_utils import label_funcs
import numpy as np

print("\nTESTING LABEL_FUNCS\n")

l1 = np.random.choice(10,size=10000)
l2 = np.tile(np.arange(10),1000)
l3 = np.tile(np.arange(20),500)
label_funcs.translate_labellings(l1,l2,subsample_size='none')
print("Should see a user warning:")
label_funcs.translate_labellings(np.array([]),np.array([]),subsample_size='none')

# Size preserving
a,b = label_funcs.get_num_labels(l3),label_funcs.get_num_labels(l2)
c = label_funcs.get_num_labels(label_funcs.translate_labellings(l3,l2))
d = label_funcs.get_num_labels(label_funcs.translate_labellings(l3,l2,preserve_sizes=True))
print(f"Should be 20,10,10,20: {int(a)},{int(b)},{int(c)},{int(d)}")

x=np.load('x.npy')
y=np.load('y.npy')
nx = label_funcs.get_num_labels(x)
ny = label_funcs.get_num_labels(y)
assert nx==6
assert ny==5

tx = label_funcs.get_num_labels(label_funcs.translate_labellings(x,y,preserve_sizes=True))
assert tx==6
tx1 = label_funcs.get_num_labels(label_funcs.translate_labellings(x,y,preserve_sizes=False))
assert tx1==5

label_funcs.get_num_labels(label_funcs.translate_labellings(l2,l3))
label_funcs.get_num_labels(label_funcs.translate_labellings(l2,l3,preserve_sizes=True))

# Unique labels
assert label_funcs.unique_labels(l1) == label_funcs.unique_labels(l2)
assert (label_funcs.compress_labels(l1+1)[0] == l1).all()
assert list(label_funcs.label_counts(l2).values())==[1000]*10
print(f"Should see a number close to 0.1:\n\t{label_funcs.accuracy(l1,l2)}")
print(f"Should see a number close to 0.1:\n\t{label_funcs.mean_f1(l1,l2)}")

for _ in range(10):
    test_num_labs = np.random.randint(2,15)
    test_size = np.random.randint(100,10000)
    x=np.random.randint(test_num_labs,size=(test_size))
    assert label_funcs.label_counts(x) == label_funcs.label_counts_without_torch(x)


babled = np.random.randint(10,size=(5,1000))
babled1 = [np.random.randint(10,size=(1000)) for _ in range(5)]
babled_wrong = np.random.randint(10,size=(5,1000,6))
label_funcs.debable(babled,pivot='none')
label_funcs.debable(babled1,pivot='none')
label_funcs.dummy_labels(7,110)
print("Should see TranslationError:")
label_funcs.debable(babled_wrong,pivot='none')
