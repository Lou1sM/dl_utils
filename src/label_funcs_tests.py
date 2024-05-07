from dl_utils import label_funcs
import numpy as np

print("\nTESTING LABEL_FUNCS\n")

l1 = np.random.choice(10,size=10000)
l2 = np.tile(np.arange(10),1000)
l3 = np.tile(np.arange(20),500)
label_funcs.translate_labellings(l1,l2,subsample_size='none')
print("Should see a user warning:")
label_funcs.translate_labellings(np.array([]),np.array([]),subsample_size='none')

def check_round_trip(orig,perfect):
    vocab = label_funcs.unique_labels(orig)
    l2_size = 2*len(vocab) if perfect else len(vocab)//2
    random_trans_dict = dict(zip(vocab,np.random.choice(range(l2_size),size=len(vocab),replace=not perfect)))
    translated = np.array([random_trans_dict[x] for x in orig])
    recovered = label_funcs.translate_labellings(translated,orig)
    if perfect: assert (recovered==orig).all()
    else: assert recovered.shape == orig.shape

def check_perfect_and_imperfect_round_trip(orig):
    check_round_trip(orig,True)
    check_round_trip(orig,False)

check_perfect_and_imperfect_round_trip(np.random.randint(10,size=(1000)))
check_perfect_and_imperfect_round_trip(np.random.randint(20,size=(1000)))
l1_vocab = np.random.choice(50,size=10,replace=False)
check_perfect_and_imperfect_round_trip(np.random.choice(l1_vocab,size=(1000)))

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
assert tx1==4

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

continuous_labels = np.array([[0.33333333, 0.55555556, 1., 0.85714286, 0.66666667, 0.71428571],
       [0.33333333, 0.55555556, 0.55555556, 1., 0.33333333, 0.42857143],
       [0.55555556, 1.        , 0.22222222, 0.14285714, 0.33333333, 0.71428571],
       [0.55555556, 0.33333333, 0.44444444, 0.57142857, 0., 0.78571429],
       [1.        , 0.77777778, 0.44444444, 0.14285714, 1., 0.],
       [0.        , 0.88888889, 0.        , 0.57142857, 0.66666667, 0.71428571]])

print(continuous_labels, '\n-->\n',label_funcs.discretize_labels(continuous_labels))

babeled = np.random.randint(10,size=(5,1000))
babeled1 = [np.random.randint(10,size=(1000)) for _ in range(5)]
babeled_wrong = np.random.randint(10,size=(5,1000,6))
label_funcs.debable(babeled,pivot='none')
label_funcs.debable(babeled1,pivot='none')
label_funcs.dummy_labels(7,110)
print("Should see TranslationError:")
label_funcs.debable(babeled_wrong,pivot='none')
