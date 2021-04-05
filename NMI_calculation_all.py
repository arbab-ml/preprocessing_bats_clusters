
import pandas as pd
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import fowlkes_mallows_score
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
            # if (i>=j):
            all_combinations=all_combinations+[(i,j)]
    return all_combinations

def calculate_NMI(first_file, second_file):
    first_results=pd.read_csv(first_file)
    first_results["Image Name"]= first_results["Image Name"].apply(lambda x: x.split('/')[-1])
    first_results=first_results[["Image Name", "Prediction"] ]
    
    second_results=pd.read_csv(second_file)
    second_results["Image Name"]= second_results["Image Name"].apply(lambda x: x.split('/')[-1])
    second_results=second_results[["Image Name", "Prediction"] ]
    
    merged_both=pd.merge(first_results, second_results, on="Image Name", how='inner', left_index=False, right_index=False)
    #label true, label predicted
    result1=normalized_mutual_info_score(merged_both["Prediction_x"], merged_both["Prediction_y"])
    result2=adjusted_rand_score(merged_both["Prediction_x"], merged_both["Prediction_y"])
    result3=adjusted_mutual_info_score(merged_both["Prediction_x"], merged_both["Prediction_y"])
    result4=completeness_score(merged_both["Prediction_x"], merged_both["Prediction_y"],)
    result=('N NMI: ' +str(result1)[:4]     +', '  +'Ad Rand: ' +str(result2)[:4]    
    +', '  +'Ad NMI: ' +str(result3)[:4]   +', '  +'Completeness: ' +str(result4)[:4]      )
    return result

files_directory="/media/rabi/Data/ThesisData/audio data analysis/audio-clustering/plots_15march_b/spectrograms_normalized/results/results_csvs/"
file_names=[ "results_iic", "results_jule", "results_SCAN","results_imsat_10_clusters"]
files_combinations=create_combinations(file_names)

index = pd.MultiIndex.from_tuples(files_combinations, names=["first", "second"])
results_df = pd.Series( index=index)
for two_files in files_combinations:
    first_file=files_directory+two_files[0]+'.csv'
    second_file=files_directory+two_files[1]+'.csv'
    scores=calculate_NMI(first_file, second_file)
    results_df[two_files]=scores
    
results_df
print(results_df)