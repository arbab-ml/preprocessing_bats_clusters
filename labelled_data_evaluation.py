


import pandas as pd
import numpy as np
from munkres import Munkres
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import mutual_info_score
from sklearn.metrics.cluster import v_measure_score


input_file="/media/rabi/Data/ThesisData/audio data analysis/specie-clustering/Identified calls/all_specie/spectrograms_vmin_vmax_highpass_balance_top5_2_cropped/results/results_deepcluster16.csv"
input_data=pd.read_csv(input_file)
input_data["id_name"]=  input_data["Image Name"].str[:].apply(lambda x: x.split('/')[-1][:6])
input_data["ground truth"] = input_data["id_name"].astype('category').cat.codes
#Now calculating F1
##########################################################################################################
def check_clusterings(labels_true, labels_pred):
    # labels_true = check_array(
    #     labels_true, ensure_2d=False, ensure_min_samples=0, dtype=None,
    # )

    # labels_pred = check_array(
    #     labels_pred, ensure_2d=False, ensure_min_samples=0, dtype=None,
    # )

    # type_label = type_of_target(labels_true)
    # type_pred = type_of_target(labels_pred)

    # if 'continuous' in (type_pred, type_label):
    #     msg = f'Clustering metrics expects discrete values but received' \
    #           f' {type_label} values for label, and {type_pred} values ' \
    #           f'for target'
    #     warnings.warn(msg, UserWarning)

    # input checks
    # if labels_true.ndim != 1:
    #     raise ValueError(
    #         "labels_true must be 1D: shape is %r" % (labels_true.shape,))
    # if labels_pred.ndim != 1:
    #     raise ValueError(
    #         "labels_pred must be 1D: shape is %r" % (labels_pred.shape,))
    # check_consistent_length(labels_true, labels_pred)

    return labels_true, labels_pred


def pair_confusion_matrix(labels_true, labels_pred):
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    n_samples = np.int64(labels_true.shape[0])

    # Computation using the contingency data
    contingency = contingency_matrix(
        labels_true, labels_pred, sparse=True
    )
    n_c = np.ravel(contingency.sum(axis=1))
    n_k = np.ravel(contingency.sum(axis=0))
    sum_squares = (contingency.data ** 2).sum()
    C = np.empty((2, 2), dtype=np.int64)
    C[1, 1] = sum_squares - n_samples
    C[0, 1] = contingency.dot(n_k).sum() - sum_squares
    C[1, 0] = contingency.transpose().dot(n_c).sum() - sum_squares
    C[0, 0] = n_samples ** 2 - C[0, 1] - C[1, 0] - sum_squares
    return C

def rand_score(labels_true, labels_pred):
    contingency = pair_confusion_matrix(labels_true, labels_pred)
    numerator = contingency.diagonal().sum()
    denominator = contingency.sum()

    if numerator == denominator or denominator == 0:
        # Special limit cases: no clustering since the data is not split;
        # or trivial clustering where each document is assigned a unique
        # cluster. These are perfect matches hence return 1.0.
        return 1.0

    return numerator / denominator



##########################################################################################################
def f_matrix(labels_pred, labels_true):
    # Calculate F1 matrix
    cont_mat = contingency_matrix(labels_pred=labels_pred, labels_true=labels_true)
    precision = normalize(cont_mat, norm='l1', axis=0)
    recall = normalize(cont_mat, norm='l1', axis=1)
    som = precision + recall
    f1 =  np.round(np.divide((2 * recall * precision), som, out=np.zeros_like(som), where=som!=0), 3)
    return f1

def f1_hungarian(f1):
    m = Munkres()
    inverse = 1 - f1
    indices = m.compute(inverse.tolist())
    fscore = sum([f1[i] for i in indices])/len(indices)
    return fscore
result=f1_hungarian(f_matrix(input_data["Prediction"], input_data["ground truth"]))
print(result)
tot_cl=8
def compute_accuracy(y_pred, y_t):
    # compute the accuracy using Hungarian algorithm
    m = Munkres()
    mat = np.zeros((tot_cl, tot_cl))
    for i in range(tot_cl):
        for j in range(tot_cl):
            mat[i][j] = np.sum(np.logical_and(y_pred == i, y_t == j))
    indexes = m.compute(-mat)

    corresp = []
    for i in range(tot_cl):
        corresp.append(indexes[i][1])

    pred_corresp = [corresp[int(predicted)] for predicted in y_pred]
    acc = np.sum(pred_corresp == y_t) / float(len(y_t))
    return acc

# A=input_data["ground truth"]
# B=input_data["Prediction"]
# resultspx.cluster.ClusterComparison(A, B)


print("Accruacy (using Hungarian algorithm)  : ", compute_accuracy(input_data["Prediction"], input_data["ground truth"]))
# print( "normalized_mutual_info_score  : ",normalized_mutual_info_score(input_data["ground truth"], input_data["Prediction"])) 
print( "Rand_score  : ",rand_score(input_data["ground truth"], input_data["Prediction"]))
print( "Adjusted Rand Score  : ",adjusted_rand_score(input_data["ground truth"], input_data["Prediction"]))
# print( "adjusted_mutual_info_score  : ",adjusted_mutual_info_score(input_data["ground truth"], input_data["Prediction"]))
# print( "completeness_score  : ",completeness_score(input_data["ground truth"], input_data["Prediction"]))
print( "fowlkes_mallows_score  : ",fowlkes_mallows_score(input_data["ground truth"], input_data["Prediction"]))
# print( "homogeneity_score  : ",homogeneity_score(input_data["ground truth"], input_data["Prediction"]))
print( "mutual_info_score  : ",mutual_info_score(input_data["ground truth"], input_data["Prediction"]))
# print( "v_measure_score  : ",v_measure_score(input_data["ground truth"], input_data["Prediction"]))


print("done")
