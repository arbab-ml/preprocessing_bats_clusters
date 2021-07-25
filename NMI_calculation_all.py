
import pandas as pd
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import mutual_info_score
from sklearn.metrics.cluster import v_measure_score
all_metrics_list=[normalized_mutual_info_score,mutual_info_score,
 adjusted_mutual_info_score, adjusted_rand_score, completeness_score, fowlkes_mallows_score, homogeneity_score, v_measure_score]

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
            if (i!=j):
                all_combinations=all_combinations+[(i,j)]
    return all_combinations

def calculate_metric(first_file, second_file, metric_name):
    first_results=pd.read_csv(first_file)
    first_results["Image Name"]= first_results["Image Name"].apply(lambda x: x.split('/')[-1])
    first_results=first_results[["Image Name", "Prediction"] ]
    
    second_results=pd.read_csv(second_file)
    second_results["Image Name"]= second_results["Image Name"].apply(lambda x: x.split('/')[-1])
    second_results=second_results[["Image Name", "Prediction"] ]
    
    merged_both=pd.merge(first_results, second_results, on="Image Name", how='inner', left_index=False, right_index=False)
    #label true, label predicted
    result=metric_name(merged_both["Prediction_x"], merged_both["Prediction_y"])
    # result2=adjusted_rand_score(merged_both["Prediction_x"], merged_both["Prediction_y"])
    # result3=adjusted_mutual_info_score(merged_both["Prediction_x"], merged_both["Prediction_y"])
    # result4=completeness_score(merged_both["Prediction_x"], merged_both["Prediction_y"],)
    # result=('N NMI: ' +str(result1)[:4]     +', '  +'Ad Rand: ' +str(result2)[:4]    
    # +', '  +'Ad NMI: ' +str(result3)[:4]   +', '  +'Completeness: ' +str(result4)[:4]      )
    return result

files_directory="/media/rabi/Data/ThesisData/audio data analysis/audio-clustering/plots_26april/spectrograms_normalized_croped_128/results/"
file_names=[ "IIC","IMSAT", "DEEPCLUSTER", "JULE", "SCAN"]  #EXCLUDED JULE FOR NOW
associated_files={"IIC": "results_iic",
"JULE": "results_jule",
"SCAN":"results_scan",
"IMSAT":"results_imsat",
"DEEPCLUSTER":"results_deepcluster"}

files_combinations=create_combinations(file_names)

index = pd.MultiIndex.from_tuples(files_combinations, names=["Algorithm-1", "Algorithm-2"])
results_df = pd.Series( index=index)
results_df_all=results_df.copy().to_frame()
for metric_name in all_metrics_list:
    results_df = pd.Series( index=index)
    for two_files in files_combinations:
        first_file=files_directory+associated_files[two_files[0]]+'.csv'
        second_file=files_directory+associated_files[two_files[1]]+'.csv'
        scores=calculate_metric(first_file, second_file, metric_name)
        results_df[two_files]=scores
        
    results_df_all=pd.concat([results_df_all, results_df.to_frame().rename(columns={0:metric_name.__name__})], axis=1)

results_df_all=results_df_all.drop(columns=[0])
# results_df_all
# print(results_df_all)
results_df_all.to_csv(files_directory+ "inter-algo scores.csv" )