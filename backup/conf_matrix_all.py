
import pandas as pd
import numpy as np
from munkres import Munkres
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import normalize


# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_score.html#sklearn.metrics.homogeneity_score
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mutual_info_score.html#sklearn.metrics.mutual_info_score
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.v_measure_score.html#sklearn.metrics.v_measure_score
##EXPLORE CONTINGENCY MATRIX
pd.options.display.width = 5000
pd.options.display.max_colwidth = 5000

def create_combinations(file_name):
    all_combinations=[]
    for i in file_name:
        for j in file_name:
            if (i>j):
                all_combinations=all_combinations+[(i,j)]
    return all_combinations

def calculate_metric(first_file, second_file):
    first_results=pd.read_csv(first_file)
    first_results["Image Name"]= first_results["Image Name"].apply(lambda x: x.split('/')[-1])
    first_results=first_results[["Image Name", "Prediction"] ]
    
    second_results=pd.read_csv(second_file)
    second_results["Image Name"]= second_results["Image Name"].apply(lambda x: x.split('/')[-1])
    second_results=second_results[["Image Name", "Prediction"] ]
    
    merged_both=pd.merge(first_results, second_results, on="Image Name", how='inner', left_index=False, right_index=False)
    #label true, label predicted
    # result=metric_name(merged_both["Prediction_x"], merged_both["Prediction_y"])
    a=pd.crosstab(merged_both.Prediction_x,merged_both.Prediction_y, normalize='index')\
    .round(4)*100

    return a

files_directory="/media/rabi/Data/ThesisData/audio data analysis/audio-clustering/plots_26april/spectrograms_normalized_croped_128/results/"
file_names=[ "IIC","IMSAT", "DEEPCLUSTER", "JULE", "SCAN", "K-Medoid"]  #EXCLUDED JULE FOR NOW
associated_files={"IIC": "results_iic",
"JULE": "results_jule",
"SCAN":"results_scan",
"IMSAT":"results_imsat",
"DEEPCLUSTER":"results_deepcluster",
"K-Medoid":"results_kmedoid"}

files_combinations=create_combinations(file_names)

index = pd.MultiIndex.from_tuples(files_combinations, names=["Algorithm-1", "Algorithm-2"])
results_df = pd.Series( index=index)
results_df_all=results_df.copy().to_frame()
with pd.ExcelWriter('output.xlsx') as writer:
    for two_files in files_combinations:
        first_file=files_directory+associated_files[two_files[0]]+'.csv'
        second_file=files_directory+associated_files[two_files[1]]+'.csv'
        scores=calculate_metric(first_file, second_file)
        # results_df[two_files]=scores
        scores.to_excel(writer, sheet_name=two_files[0]+'-'+two_files[1])


# # results_df_all
# # print(results_df_all)
# results_df_all.to_csv(files_directory+ "inter-algo scores4.csv" )