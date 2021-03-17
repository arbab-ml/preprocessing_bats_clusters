
import pandas as pd
from sklearn.metrics.cluster import normalized_mutual_info_score


#read the results of kaleidoscope and segnemnt useful columns
kal_results=pd.read_csv("/media/rabi/Data/ThesisData/audio data analysis/audio-clustering/plots_15march/cluster_kaleidoscope.csv")
kal_results["IN FILE"] =  kal_results["IN FILE"]+' at '+(kal_results["OFFSET"]).astype(str) +'.png'


kal_results=kal_results[["IN FILE", "TOP1MATCH*"]]
kal_results["TOP1MATCH*"]=kal_results["TOP1MATCH*"].apply(lambda x: x.split('.')[1])
#kal_results["IN FILE"]=kal_results["IN FILE"].apply(lambda x: x.split('.')[0])
#read the results of ALGORITHM ans segment useful columns
algo_results=pd.read_csv("/media/rabi/Data/ThesisData/audio data analysis/audio-clustering/plots_15march/spectrograms/results/clusters_15march.csv")
algo_results["IN FILE"]= algo_results["file_directory"].apply(lambda x: x.split('/')[-1])
algo_results=algo_results[["IN FILE", "cluster"] ]

merged_both=pd.merge(kal_results, algo_results, on="IN FILE", how='inner', left_index=False, right_index=False)
#label true, label predicted
result=normalized_mutual_info_score(merged_both["TOP1MATCH*"], merged_both["cluster"],)
#Left join Kaleidoscope results with ALGORITHMS results
print(result)
#calculate the NMI
