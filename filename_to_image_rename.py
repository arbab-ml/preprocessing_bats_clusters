import pandas as pd
import os
import shutil
input_data_directory="/content/drive/MyDrive/thesisdata/whole dataset/plots_26april/spectrograms_normalized/batsnet_train/1"
input_file="/content/drive/MyDrive/thesisdata/whole dataset/plots_26april/spectrograms_normalized/results/results_jule.csv"
output_data_directory="/content/drive/MyDrive/thesisdata/whole dataset/plots_26april/spectrograms_normalized/results/results_jule"


input_data=pd.read_csv(input_file)

#further processing for kaleidoscope


counter=0
for index, (_, file_path, cluster_assigned) in input_data.iterrows():
    file_name=file_path.split('/')[-1]
    copy_this_file=os.path.join(input_data_directory, file_name)
    output_data_directory_local=output_data_directory+'/Cluster_'+ str(cluster_assigned)
    os.makedirs(output_data_directory_local, exist_ok=True)
    to_this_path=os.path.join(output_data_directory_local, (str(cluster_assigned)+ "-Cluster:"+file_name)) 
    shutil.copy2(copy_this_file,to_this_path )
    counter=counter+1
    if not (counter%50):
      print(counter)
