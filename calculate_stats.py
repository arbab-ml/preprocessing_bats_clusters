#from path /media/rabi/Data/ThesisData/Bats audio records/2019/SM4BAT/WWNP/190218-190306
import pandas as pd
import librosa
from tqdm import tqdm
from pathlib import Path
input_directory = r'/media/rabi/Data/ThesisData/Bats audio records/2019/SM4BAT/WWNP/190218-190306'
all_stats=pd.DataFrame(columns=["file_name", "length", "sample rate", "file_size"])

counter=0
for path in tqdm(Path(input_directory).rglob('*.wav')):
    save_to= str(path.relative_to(input_directory)).split('/')[-1]
    signal, sr = librosa.load(path,sr=None)    #Explicitly Setting sr=None ensures original sampling preserved -- STOVF    
    #Segmenting to a random value of 1 seconds (for initial experiemtation)
    length= signal.shape[0]/sr 
    temp_results={"file_name":save_to, "length": length, "sample rate": sr, "file_size": path.stat().st_size/1000000 }    
    all_stats=all_stats.append(temp_results, ignore_index=True)
    counter=counter+1
    
all_stats.to_csv("SM4BAT-WWNP-190218-190306_all_stats.csv")

#Total number of recordings
print("Total number of recordings: "+str(all_stats.shape[0]))

#Mean and Std length 
print("Mean length:", all_stats["length"].mean())
print("Std length:", all_stats["length"].std())
#Mean and Std file size
print("Mean size:", all_stats["file_size"].mean())
print("Std size:", all_stats["file_size"].std())