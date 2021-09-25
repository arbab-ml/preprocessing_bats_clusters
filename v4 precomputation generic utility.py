#this utility populate a csv file from the data directory - that's compatible with v4 precomputation generic.py
import os
import pandas as pd
import numpy as np 




length_of_each_audio_file=3.0 #seconds

ip_dir = r'/media/rabi/Data/ThesisData/audio data analysis/class-clustering/audio_cleaned_3_seconds'
save_folder="/media/rabi/Data/ThesisData/audio data analysis/class-clustering/plots_25sep"
populated_df=pd.DataFrame(columns=['file_name', 'detection_time', 'detection_prob'])

matches=[]
for root, dirnames, filenames in os.walk(ip_dir):
    for filename in filenames:
        if filename.lower().endswith('.wav'):
            matches.append( '/'+filename)

populated_df["file_name"]=matches

populated_df['detection_time']=length_of_each_audio_file/2
populated_df['detection_prob']=1#Since data is already curated manually, set it to 1. 
populated_df.to_csv(save_folder+'/nonbatdetect_25sep.csv')