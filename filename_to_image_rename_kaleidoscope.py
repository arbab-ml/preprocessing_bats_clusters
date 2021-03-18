

import pandas as pd
import os
import shutil
input_data_directory="/media/rabi/Data/ThesisData/audio data analysis/audio-clustering/plots_15march_b/spectrograms_normalized/batsnet_train/1"
input_file="/media/rabi/Data/ThesisData/audio data analysis/audio-clustering/plots_15march_b/spectrograms_normalized/results/cluster_kaleidoscope.csv"
output_data_directory="/media/rabi/Data/ThesisData/audio data analysis/audio-clustering/plots_15march_b/spectrograms_normalized/results/cluster_kaleidoscope"


input_data_raw=pd.read_csv(input_file)
input_data=input_data_raw[[ "IN FILE", "OFFSET","TOP1MATCH*"] ]

#further processing for kaleidoscope



for index, (file_initial, offset, cluster_assigned) in input_data.iterrows():
    cluster_assigned=int(cluster_assigned.split('.')[-1])
    file_name=file_initial+' at '+str(offset)+'.png'

    copy_this_file=os.path.join(input_data_directory, file_name)
    output_data_directory_local=output_data_directory+'/Cluster_'+ str(cluster_assigned)
    os.makedirs(output_data_directory_local, exist_ok=True)
    to_this_path=os.path.join(output_data_directory_local, (str(cluster_assigned)+ "-Cluster:"+file_name)) 
    shutil.copy2(copy_this_file,to_this_path )