import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from sklearn.metrics import normalized_mutual_info_score as mi_func
from pdb import set_trace
import gc
import math
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import torch.multiprocessing as mp
from dl_utils.tensor_funcs import numpyify


def reload():
    import importlib, utils
    importlib.reload(utils)

def save_and_check(enc,dec,fname):
    torch.save({'enc': enc, 'dec': dec},fname)
    loaded = torch.load(fname)
    e,d = loaded['enc'], loaded['dec']
    test_mods_eq(e,enc); test_mods_eq(d,dec)

def show_xb(xb): plt.imshow(xb[0,0]); plt.show()
def get_datetime_stamp(): return str(datetime.now()).split()[0][5:] + '_'+str(datetime.now().time()).split()[0][:-7]

def get_user_yesno_answer(question):
    answer = input(question+'(y/n)')
    if answer == 'y': return True
    elif answer == 'n': return False
    else:
        print("Please answer 'y' or 'n'")
        return(get_user_yesno_answer(question))

def set_experiment_dir(exp_name, overwrite):
    exp_name = get_datetime_stamp() if exp_name == "" else exp_name
    exp_dir = f'../experiments/{exp_name}'
    if not os.path.isdir(exp_dir): os.makedirs(exp_dir)
    elif exp_name.startswith('try') or overwrite: pass
    elif not get_user_yesno_answer(f'An experiment with name {exp_name} has already been run, do you want to overwrite?'):
        print('Please rerun command with a different experiment name')
        sys.exit()
    return exp_dir
def test_mods_eq(m1,m2):
    for a,b in zip(m1.parameters(),m2.parameters()):
        assert (a==b).all()

def compose(funcs):
    def _comp(x):
        for f in funcs: x=f(x)
        return x
    return _comp

def scatter_clusters(embeddings,labels,show=False):
    fig, ax = plt.subplots()
    palette = ['r','k','y','g','b','m','purple','brown','c','orange','thistle','lightseagreen','sienna']
    labels = numpyify([0]*len(embeddings)) if labels is None else numpyify(labels)
    palette = cm.rainbow(np.linspace(0,1,len(set(labels))))
    for i,label in enumerate(list(set(labels))):
        ax.scatter(embeddings[labels==label,0], embeddings[labels==label,1], s=0.2, c=[palette[i]], label=i)
    ax.legend()
    if show: plt.show()
    return ax

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def compute_multihots(l,probs):
    assert len(l) > 0
    mold = np.expand_dims(np.arange(l.max()+1),0) # (num_aes, num_labels)
    hits = (mold==np.expand_dims(l,2)) # (num_aes, dset_size, num_labels)
    if probs != 'none': hits = np.expand_dims(probs,2)*hits
    multihots = hits.sum(axis=0) # (dset_size, num_labels)
    return multihots

def n_digitify(number,num_digits):
    with_zeros = "0"*(num_digits-len(str(number))) + str(number)
    assert len(with_zeros) == num_digits and int(with_zeros) == number
    return with_zeros

def check_latents(dec,latents,show,stacked):
    _, axes = plt.subplots(6,2,figsize=(7,7))
    for i,latent in enumerate(latents):
        try:
            outimg = dec(latent[None,:,None,None])
            if stacked: outimg = outimg[-1]
            axes.flatten()[i].imshow(outimg[0,0])
        except: set_trace()
    if show: plt.show()
    plt.clf()

def dictify_list(x,key):
    assert isinstance(x,list)
    assert len(x) > 0
    assert isinstance(x[0],dict)
    return {item[key]: item for item in x}

def cont_factorial(x): return (x/np.e)**x*(2*np.pi*x)**(1/2)*(1+1/(12*x))
def cont_choose(ks): return cont_factorial(np.sum(ks))/np.prod([cont_factorial(k) for k in ks if k > 0])
def prob_results_given_c(results,cluster,prior_correct):
    """For single dpoint, prob of these results given right answer for cluster.
    ARGS:
        results (np.array): votes for this dpoint
        cluster (int): right answer to condition on
        prior_correct (\in (0,1)): guess for acc of each element of ensemble
        """

    assert len(results.shape) <= 1
    prob = 1
    results_normed = np.array(results)
    results_normed = results_normed / np.sum(results_normed)
    for c,r in enumerate(results_normed):
        if c==cluster: prob_of_result = prior_correct**r
        else: prob_of_result = ((1-prior_correct)/results.shape[0])**r
        prob *= prob_of_result
    partitions = cont_choose(results_normed)
    prob *= partitions
    try:assert prob <= 1
    except:set_trace()
    return prob

def prior_for_results(results,prior_correct):
    probs = [prob_results_given_c(results,c,prior_correct) for c in range(results.shape[0])]
    set_trace()
    return sum(probs)

def all_conditionals(results,prior_correct):
    """For each class, prob of results given that class."""
    cond_probs = [prob_results_given_c(results,c,prior_correct) for c in range(len(results))]
    assert np.sum(cond_probs) < 1.01
    return np.array(cond_probs)

def posteriors(results,prior_correct):
    """Bayes to get prob of each class given these results."""
    conditionals = all_conditionals(results,prior_correct)
    posterior_array = conditionals/np.sum(conditionals)
    return posterior_array

def posterior_corrects(results):
    probs = []
    for p in np.linspace(0.6,1.0,10):
        conditional_prob = np.prod([np.sum(all_conditionals(r,p)) for r in results])
        probs.append(conditional_prob)
    probs = np.array(probs)
    posterior_for_accs = 0.1*probs/np.sum(probs) # Prior was uniform over all accs in range
    assert posterior_for_accs.max() < 1.01
    return posterior_for_accs

def votes_to_probs(multihots,prior_correct):
    """For each dpoint, compute probs for each class, given these ensemble votes.
    ARGS:
        multihots (np.array): votes for each dpoint, size N x num_classes
        prior_correct (\in (0,1)): guess for acc of each element of ensemble
        """

    probs_list = [np.ones(multihots.shape[-1])/multihots.shape[-1] if r.max() == 0 else posteriors(r,prior_correct) for r in multihots]
    probs_array = np.array(probs_list)
    return probs_array

def check_dir(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)

def torch_save(checkpoint,directory,fname):
    check_dir(directory)
    torch.save(checkpoint,os.path.join(directory,fname))

def np_save(array,directory,fname,verbose=False):
    check_dir(directory)
    save_path = os.path.join(directory,fname)
    if verbose: print('Saving to', save_path)
    np.save(save_path,array)

def np_savez(data_dict,directory,fname):
    check_dir(directory)
    np.savez(os.path.join(directory,fname),**data_dict)

def apply_maybe_multiproc(func,input_list,split,single):
    if single:
        output_list = [func(item) for item in input_list]
    else:
        list_of_lists = []
        ctx = mp.get_context("spawn")
        num_splits = math.ceil(len(input_list)/split)
        for i in range(num_splits):
            with ctx.Pool(processes=split) as pool:
                new_list = pool.map(func, input_list[split*i:split*(i+1)])
            if num_splits != 1: print(f'finished {i}th split section')
            list_of_lists.append(new_list)
        output_list = [item for sublist in list_of_lists for item in sublist]
    return output_list

def show_gpu_memory():
    mem_used = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or hasattr(obj, 'data') and torch.is_tensor(obj.data):
                mem_used += obj.element_size() * obj.nelement()
        except: pass
    print(f"GPU memory usage: {mem_used}")

def rmi_func(pred,gt): return round(mi_func(pred,gt),4)
