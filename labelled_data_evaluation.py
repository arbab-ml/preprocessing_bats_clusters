
import pandas as pd
import numpy as np
from munkres import Munkres
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import normalize

input_file="/media/rabi/Data/ThesisData/audio data analysis/specie-clustering/Identified calls/all_specie/spectrograms_augmented_normalization_cropped/results/results_imsat2.csv"
input_data=pd.read_csv(input_file)
input_data["id_name"]=  input_data["Image Name"].str[:].apply(lambda x: x.split('/')[-1][:6])
input_data["ground truth"] = input_data["id_name"].astype('category').cat.codes
#Now calculating F1

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

result=compute_accuracy(input_data["Prediction"], input_data["ground truth"])
print(result)
print("done")
