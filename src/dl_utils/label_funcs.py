import numpy as np
import torch
import warnings
from pdb import set_trace
from scipy.optimize import linear_sum_assignment
from dl_utils.tensor_funcs import numpyify


class TranslationError(Exception):
    pass

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
    if len(labels1) != len(labels2):
        raise TranslationError(f"len labels1 {len(labels1)} must equal len labels2 {len(labels2)}")
    return len([idx for idx in range(len(labels2)) if labels1[idx]==label1 and labels2[idx] != label2])

def get_trans_dict(trans_from_labels,trans_to_labels,subsample_size='none'):
    # First compress each labelling, retain compression dicts
    trans_from_labels, tdf, _ = compress_labels(trans_from_labels)
    trans_to_labels, tdt, _ = compress_labels(trans_to_labels)
    reverse_tdf = {v:k for k,v in tdf.items()}
    reverse_tdt = {v:k for k,v in tdt.items()}
    if trans_from_labels.shape != trans_to_labels.shape:
        raise TranslationError(f"trans_to_labels: {trans_to_labels.shape} doesn't equal trans_to_labels shape: {trans_from_labels.shape}")
    num_from_labs = get_num_labels(trans_from_labels)
    num_to_labs = get_num_labels(trans_to_labels)
    if subsample_size != 'none':
        if trans_from_labels.shape != trans_to_labels.shape:
            raise TranslationError(f"subsample_size is too small, it must be at least min of the number of different from labels and the number of different to labels, which in this case are {num_from_labs} and {num_to_labs}")
        subsample_size = min(len(trans_from_labels),subsample_size)
    if num_from_labs <= num_to_labs:
        trans_dict = get_fanout_trans_dict(trans_from_labels,trans_to_labels,subsample_size)
        leftovers = np.array([x for x in unique_labels(trans_to_labels) if x not in trans_dict.values()])
    else:
        trans_dict,leftovers = get_fanin_trans_dict(trans_from_labels,trans_to_labels,subsample_size)
    # Account for the possible changes in the above compression
    trans_dict = {reverse_tdf[k]:reverse_tdt[v] for k,v in trans_dict.items()}
    return trans_dict,leftovers

def translate_labellings(trans_from_labels,trans_to_labels,subsample_size='none'):
    if trans_from_labels.shape != trans_to_labels.shape:
        raise TranslationError(f"trans_to_labels: {trans_to_labels.shape} doesn't equal trans_to_labels shape: {trans_from_labels.shape}")
    if len(trans_from_labels) == 0:
        warnings.warn("You're translating an empty labelling")
        return trans_from_labels
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
    if len(col_ind) != len(set(trans_from_labels[trans_from_labels != -1])):
        raise TranslationError(f"then translation cost matrix is the wrong size for some reason")
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
            if unique_labels(trans_from_labels_subsample) == unique_trans_from_labels and unique_labels(trans_to_labels_subsample) == unique_trans_to_labels: break
        sample_indices = np.random.choice(range(trans_from_labels.shape[0]),subsample_size,replace=False)
        trans_from_labels_subsample = trans_from_labels[sample_indices]
        trans_to_labels_subsample = trans_to_labels[sample_indices]
        cost_matrix = np.array([[label_assignment_cost(trans_to_labels_subsample,trans_from_labels_subsample,l1,l2) for l2 in unique_trans_from_labels if l2 != -1] for l1 in unique_trans_to_labels if l1 != -1])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    if len(col_ind) != get_num_labels(trans_to_labels):
        raise TranslationError(f"then translation cost matrix is the wrong size for some reason")
    cl = col_ind.tolist()
    trans_dict = {f:cl.index(f) for f in cl}
    while True:
        untranslated = [i for i in unique_trans_from_labels if i not in trans_dict.keys()]
        if len(untranslated) == 0: break
        unique_untranslated = unique_labels(untranslated)
        # Now assign the additional, unassigned items
        cost_matrix2 = np.array([[label_assignment_cost(trans_from_labels,trans_to_labels,l1,l2) for l2 in unique_trans_to_labels if l2 != -1] for l1 in unique_untranslated if l1 != -1])
        row_ind2, col_ind2 = linear_sum_assignment(cost_matrix2)
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

def debable(labellings,pivot,subsample_size='none'):
    #labellings_list.sort(key=lambda x: x.max())
    if isinstance(labellings,np.ndarray):
        if labellings.ndim != 2:
            raise TranslationError(f"If debabling array, it should have 2 dimensions, but here it has {labellings.ndim}")
        labellings_list = [r for r in labellings]
    else:
        labellings_list = labellings
    if pivot == 'none':
        pivot = labellings_list.pop(0)
        translated_list = [pivot]
    else:
        translated_list = []
    for not_lar in labellings_list:
        not_lar_translated = translate_labellings(not_lar,pivot,subsample_size=subsample_size)
        translated_list.append(not_lar_translated)
    return translated_list

def accuracy(labels1,labels2,subsample_size='none',precision=4):
    if labels1.shape != labels2.shape:
        raise TranslationError(f"labels1: {labels1.shape} doesn't equal labels2 shape: {labels2.shape}")
    if len(labels1) == 0:
        warnings.warn("You're translating an empty labelling")
        return 0
    trans_labels = translate_labellings(labels1,labels2,subsample_size)
    return round(sum(trans_labels==numpyify(labels2))/len(labels1),precision)

def f1(bin_classifs_pred,bin_classifs_gt,precision=4):
    tp = sum(bin_classifs_pred*bin_classifs_gt)
    if tp==0: return 0
    fp = sum(bin_classifs_pred*~bin_classifs_gt)
    fn = sum(~bin_classifs_pred*bin_classifs_gt)

    prec = tp/(tp+fp)
    rec = tp/(tp+fn)
    return round((2*prec*rec)/(prec+rec),precision)

def mean_f1(labels1,labels2,subsample_size='none',precision=4):
    trans_labels = translate_labellings(labels1,labels2,subsample_size)
    lab_f1s = []
    for lab in unique_labels(trans_labels):
        lab_booleans1 = trans_labels==lab
        lab_booleans2 = labels2==lab
        lab_f1s.append(f1(lab_booleans1,lab_booleans2,precision=15))
    return round(sum(lab_f1s)/len(lab_f1s),precision)

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

def dummy_labels(num_classes,size):
    main_chunk = np.tile(np.arange(num_classes),size//num_classes)
    extra_chunk = np.arange(num_classes)[:size%num_classes]
    combined = np.concatenate((main_chunk,extra_chunk), axis=0)
    assert combined.shape[0] == size
    return combined
