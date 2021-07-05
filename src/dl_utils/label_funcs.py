import numpy as np
import torch
from pdb import set_trace
from scipy.optimize import linear_sum_assignment
from dl_utils.tensor_funcs import numpyify


def unique_labels(labels):
    if isinstance(labels,np.ndarray) or isinstance(labels,list):
        return set(labels)
    elif isinstance(labels,torch.Tensor):
        unique_tensor = labels.unique()
        return set(unique_tensor.tolist())
    else:
        print("Unrecognized type for labels:", type(labels))
        raise TypeError
def label_assignment_cost(labels1,labels2,label1,label2):
    assert len(labels1) == len(labels2), f"len labels1 {len(labels1)} must equal len labels2 {len(labels2)}"
    return len([idx for idx in range(len(labels2)) if labels1[idx]==label1 and labels2[idx] != label2])

def get_trans_dict(trans_from_labels,trans_to_labels,subsample_size):
    # First compress each labelling, retain compression dicts
    trans_from_labels, tdf, _ = compress_labels(trans_from_labels)
    trans_to_labels, tdt, _ = compress_labels(trans_to_labels)
    reverse_tdf = {v:k for k,v in tdf.items()}
    reverse_tdt = {v:k for k,v in tdt.items()}
    assert trans_from_labels.shape == trans_to_labels.shape
    num_from_labs = get_num_labels(trans_from_labels)
    num_to_labs = get_num_labels(trans_to_labels)
    assert subsample_size == 'none' or subsample_size > min(num_from_labs,num_to_labs)
    if num_from_labs <= num_to_labs:
        trans_dict = get_fanout_trans_dict(trans_from_labels,trans_to_labels,subsample_size)
        leftovers = np.array([x for x in unique_labels(trans_to_labels) if x not in trans_dict.values()])
    else:
        trans_dict,leftovers = get_fanin_trans_dict(trans_from_labels,trans_to_labels,subsample_size)
    # Account for the possible changes in the above compression
    trans_dict = {reverse_tdf[k]:reverse_tdt[v] for k,v in trans_dict.items()}
    return trans_dict,leftovers

def translate_labellings(trans_from_labels,trans_to_labels,subsample_size):
    trans_dict, leftovers = get_trans_dict(trans_from_labels,trans_to_labels,subsample_size)
    return np.array([trans_dict[l] for l in trans_from_labels])

def get_fanout_trans_dict(trans_from_labels,trans_to_labels,subsample_size):
    unique_trans_from_labels = unique_labels(trans_from_labels)
    unique_trans_to_labels = unique_labels(trans_to_labels)
    if subsample_size == 'none':
        cost_matrix = np.array([[label_assignment_cost(trans_from_labels,trans_to_labels,l1,l2) for l2 in unique_trans_to_labels if l2 != -1] for l1 in unique_trans_from_labels if l1 != -1])
    else:
        num_trys = 0
        while True:
            num_trys += 1
            if num_trys == 5: set_trace()
            sample_indices = np.random.choice(range(trans_from_labels.shape[0]),subsample_size,replace=False)
            trans_from_labels_subsample = trans_from_labels[sample_indices]
            trans_to_labels_subsample = trans_to_labels[sample_indices]
            if unique_labels(trans_from_labels_subsample) == unique_trans_from_labels and unique_labels(trans_to_labels_subsample) == unique_trans_to_labels: break
        cost_matrix = np.array([[label_assignment_cost(trans_from_labels_subsample,trans_to_labels_subsample,l1,l2) for l2 in unique_trans_from_labels if l2 != -1] for l1 in unique_trans_from_labels if l1 != -1])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    try: assert len(col_ind) == len(set(trans_from_labels[trans_from_labels != -1]))
    except: set_trace()
    trans_dict = {l:col_ind[l] for l in unique_labels(trans_from_labels)}
    trans_dict[-1] = -1
    return trans_dict

def get_fanin_trans_dict(trans_from_labels,trans_to_labels,subsample_size):
    unique_trans_from_labels = unique_labels(trans_from_labels)
    unique_trans_to_labels = unique_labels(trans_to_labels)
    if subsample_size == 'none':
        cost_matrix = np.array([[label_assignment_cost(trans_to_labels,trans_from_labels,l1,l2) for l2 in unique_trans_from_labels if l2 != -1] for l1 in unique_trans_to_labels if l1 != -1])
    else:
        while True: # Keep trying random indices unitl you reach one that contains all labels
            sample_indices = np.random.choice(range(trans_from_labels.shape[0]),subsample_size,replace=False)
            trans_from_labels_subsample = trans_from_labels[sample_indices]
            trans_to_labels_subsample = trans_to_labels[sample_indices]
            if unique_labels(trans_from_labels_subsample) == unique_trans_from_labels and unique_labels(trans_from_labels_subsample) == unique_trans_to_labels: break
        sample_indices = np.random.choice(range(trans_from_labels.shape[0]),subsample_size,replace=False)
        trans_from_labels_subsample = trans_from_labels[sample_indices]
        trans_to_labels_subsample = trans_to_labels[sample_indices]
        cost_matrix = np.array([[label_assignment_cost(trans_to_labels_subsample,trans_from_labels_subsample,l1,l2) for l2 in unique_trans_from_labels if l2 != -1] for l1 in unique_trans_to_labels if l1 != -1])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assert len(col_ind) == get_num_labels(trans_to_labels)
    untranslated = [i for i in range(cost_matrix.shape[1]) if i not in col_ind]
    unique_untranslated = unique_labels(untranslated)
    # Now assign the additional, unassigned items
    cost_matrix2 = np.array([[label_assignment_cost(trans_from_labels,trans_to_labels,l1,l2) for l2 in unique_trans_to_labels if l2 != -1] for l1 in unique_untranslated if l1 != -1])
    row_ind2, col_ind2 = linear_sum_assignment(cost_matrix2)
    cl = col_ind.tolist()
    trans_dict = {f:cl.index(f) for f in cl}
    for u,t in zip(untranslated,col_ind2): trans_dict[u]=t
    trans_dict[-1] = -1
    return trans_dict, unique_untranslated

def get_confusion_mat(labels1,labels2):
    if max(labels1) != max(labels2):
        print('Different numbers of clusters, no point trying'); return
    trans_labels = translate_labellings(labels1,labels2)
    num_labels = max(labels1)+1
    confusion_matrix = np.array([[len([idx for idx in range(len(labels2)) if labels1[idx]==l1 and labels2[idx]==l2]) for l2 in range(num_labels)] for l1 in range(num_labels)])
    confusion_matrix = confusion_matrix[:,trans_labels]
    idx = np.arange(num_labels)
    confusion_matrix[idx,idx]=0
    return confusion_matrix

def debable(labellings_list,pivot):
    labellings_list.sort(key=lambda x: x.max())
    if pivot == 'none':
        pivot = labellings_list.pop(0)
        translated_list = [pivot]
    else:
        translated_list = []
    for not_lar in labellings_list:
        not_lar_translated = translate_labellings(not_lar,pivot)
        translated_list.append(not_lar_translated)
    return translated_list

def accuracy(labels1,labels2,subsample_size='none'):
    trans_labels = translate_labellings(labels1,labels2,subsample_size)
    return sum(trans_labels==numpyify(labels2))/len(labels1)

def f1(bin_classifs_pred,bin_classifs_gt):
    tp = sum(bin_classifs_pred*bin_classifs_gt)
    if tp==0: return 0
    fp = sum(bin_classifs_pred*~bin_classifs_gt)
    fn = sum(~bin_classifs_pred*bin_classifs_gt)

    prec = tp/(tp+fp)
    rec = tp/(tp+fn)
    return (2*prec*rec)/(prec+rec)

def mean_f1(labels1,labels2):
    subsample_size = min(len(labels1),30000)
    trans_labels = translate_labellings(labels1,labels2,subsample_size)
    lab_f1s = []
    for lab in unique_labels(trans_labels):
        lab_booleans1 = trans_labels==lab
        lab_booleans2 = labels2==lab
        lab_f1s.append(f1(lab_booleans1,lab_booleans2))
    return sum(lab_f1s)/len(lab_f1s)

def compress_labels(labels):
    if isinstance(labels,torch.Tensor): labels = labels.detach().cpu().numpy()
    x = sorted([lab for lab in set(labels) if lab != -1])
    trans_dict = {lab:x.index(lab) for lab in set(labels) if lab != -1}
    trans_dict[-1] = -1
    new_labels = np.array([trans_dict[lab] for lab in labels])
    changed = any([k!=v for k,v in trans_dict.items()])
    return new_labels,trans_dict,changed

def get_num_labels(labels):
    assert labels.ndim == 1
    return len([lab for lab in unique_labels(labels) if lab != -1])

def label_counts(labels):
    assert labels.ndim == 1
    return {x:sum(labels==x) for x in unique_labels(labels)}
