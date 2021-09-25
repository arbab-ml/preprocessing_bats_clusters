import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa
import pandas as pd
import numpy as np
import librosa.display
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy import stats
import soundfile as sf
import os
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
from scipy.io.wavfile import read as read_wav
import array
import glob
from pathlib import Path
plt.ioff()#turning interactive plotting OFF
import cv2
import IPython.display as ipd
from pydub import AudioSegment

#Reference: https://www.kaggle.com/haqishen/augmentation-methods-for-audio
#Reference: https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53

def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

EPS = 1e-8

def calc_fft(y, rate):
	n = len(y)
	freq = np.fft.rfftfreq(n, d=1/rate)
	Y = abs(np.fft.rfft(y)/n)
	return Y, freq

def plot_spectrogram(signals, labels_flag=False, rate=44100,mode="simple", target_length=3):
	bottom_value=0#20000
	top_value=11000#92000
	fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False,
								 sharey=True, figsize=(5,5))
	fig.suptitle('Spectrogram', size=16)
	hop_length_value=512
	n_fft_value=5120
	if (mode=="simple"):
		spec = np.abs(librosa.stft(signals, hop_length=hop_length_value, n_fft=n_fft_value))
		spec = (librosa.amplitude_to_db(spec, ref=np.max))  #_normalize removed
		axes=librosa.display.specshow(spec, sr=rate,hop_length=hop_length_value, x_axis='time', y_axis='linear') #when computing an STFT, you can pass that same parameter to specshow. 
																					#This ensures that axis scales (e.g. time or frequency) are computed correctly.
		ax2 = fig.gca() #getting this to set the range
		ax2.set_ylim(bottom=bottom_value, top=top_value) # Setting frequency 
		return fig

	elif (mode == "timeshift"):
		start_ = int(np.random.uniform(- (0.2*rate),(0.2*rate)))
		#print('time shift: ',start_)
		if start_ >= 0:
			wav_time_shift = np.r_[signals[start_:], np.random.choice(signals, start_ )]   ##padding empty space with elelments of same distribution
			
		else:
			wav_time_shift = np.r_[np.random.choice(signals, -start_ ), signals[:start_]]
			
		#^ processing done, now plotting
		spec = np.abs(librosa.stft(wav_time_shift, hop_length=hop_length_value, n_fft=n_fft_value))
		spec = (librosa.amplitude_to_db(spec, ref=np.max))#_normalize removed
		axes=librosa.display.specshow(spec, sr=rate, x_axis='time',hop_length=hop_length_value, y_axis='linear')
		ax2 = fig.gca() #getting this to set the range
		#print("Before adjusting axes, the yaxis range was: ", ax2.get_ylim())
		ax2.set_ylim(bottom=bottom_value, top=top_value) # Setting frequency betweei 8KHz and 92KHz
		return fig

	elif (mode == "speedtune"):
		speed_rate = np.random.uniform(0.7,1.3)
		wav_speed_tune = cv2.resize(signals, (1, int(len(signals) * speed_rate))).squeeze()
		#print('speed rate: %.3f' % speed_rate, '(lower is faster)')
		if len(wav_speed_tune) < (target_length*rate):
			pad_len = (target_length*rate) - len(wav_speed_tune)
			wav_speed_tune = np.r_[np.random.choice(signals,int(pad_len/2)),
								wav_speed_tune,
								np.random.choice(signals,int(np.ceil(pad_len/2)))]
		else: 
			cut_len = len(wav_speed_tune) - (target_length*rate)
			wav_speed_tune = wav_speed_tune[int(cut_len/2):int(cut_len/2)+int(target_length*rate)]
		#^ processing done, now plotting
		spec = np.abs(librosa.stft(wav_speed_tune, hop_length=hop_length_value, n_fft=n_fft_value))
		spec = (librosa.amplitude_to_db(spec, ref=np.max))#_normalize removed
		axes=librosa.display.specshow(spec, sr=rate, x_axis='time',hop_length=hop_length_value, y_axis='linear')
		ax2 = fig.gca() #getting this to set the range
		#print("Before adjusting axes, the yaxis range was: ", ax2.get_ylim())
		ax2.set_ylim(bottom=bottom_value, top=top_value) # Setting frequency betweei 8KHz and 92KHz
		return fig
	else:
		print("Wrong Mode Selected")

	

mode="plotting"
target_sampling_rate=22000
sample_size=float('inf')
required_output_number=float('inf')
input_directory = r'/media/rabi/Data/ThesisData/audio data analysis/class-clustering/audio_cleaned_3_seconds'
bat_calls_data_dir="/media/rabi/Data/ThesisData/audio data analysis/class-clustering/plots_25sep/nonbatdetect_25sep.csv"
output_directory=r'/media/rabi/Data/ThesisData/audio data analysis/class-clustering/plots_25sep/activity_spectrograms_normalized_25sep_balanced'
target_dBFS=0  # Used for Normalization
final_target_length=3.0 # second
normalize_samples=False #Only true for batdetect version
iterator=0
output_number_iterator=0
label_offset=9  #Number of characters in start of label
total_categories=4
is_data_labelled=True
random.seed(5) 
all_stats=pd.DataFrame(columns=["file_name", "length", "sample rate"])

bat_calls_data_raw=pd.read_csv(bat_calls_data_dir)
bat_calls_data=bat_calls_data_raw[ bat_calls_data_raw["detection_prob"]>=0.95].reset_index(drop=True)

#selecting maximum probabiliyt value [deprecated]
# bat_calls_data_processed=bat_calls_data.iloc[bat_calls_data.groupby('file_name')['detection_prob'].agg(pd.Series.idxmax)]

#selecting from random place (given: probablity there is >=0.95).
# The line below will only impact the batdetect output file. 

if (is_data_labelled):
	#####balancing the numbers from each  AND selecting top x
	bat_calls_data["class"]=bat_calls_data["file_name"].str[1:label_offset+1]
	bat_calls_data = bat_calls_data.groupby('class')
	target_count=bat_calls_data.size().sort_values(ascending=False)[total_categories-1]  #TAKING TOP 5 CATEGORIES

	bat_calls_data=(bat_calls_data
					.filter(lambda x: len(x) >= target_count)
					.groupby('class')
					.apply(lambda x: x.sample(target_count, random_state=1))
					.reset_index(drop=True))

	bat_calls_data_processed=bat_calls_data   #SELECTING ALL OF THE BAT CALLS IN A FILE

else:
	bat_calls_data_processed=bat_calls_data.groupby('file_name').agg(pd.DataFrame.sample).reset_index()



for index, row_data in tqdm(bat_calls_data_processed.iterrows()):
	try:
		#giving CPU some rest
		if (output_number_iterator%100==0):
			print("Resting .. ")
			time.sleep(10)

		path=input_directory + row_data["file_name"]
		save_to=row_data["file_name"].split('/')[-1]  +' at '+str(row_data["detection_time"])
		#save_to= str(path.relative_to(input_directory)).split('/')[-1]
		
		#OLD METHOD--> signal, sr = librosa.load(path,sr=None)    #Explicitly Setting sr=None ensures original sampling preserved -- STOVF   

		sound = AudioSegment.from_file(path)

		if (sound.frame_rate!=target_sampling_rate):
			sound=sound.set_frame_rate(target_sampling_rate)
		
		if (normalize_samples==True):
			sound=match_target_amplitude(sound, target_dBFS)
		
		samples = sound.get_array_of_samples()
		new_sound = sound._spawn(samples)
		signal = np.array(samples).astype(np.float32)
		sr=sound.frame_rate
 

		#Segmenting to a random value of x seconds (for initial experiemtation)
		length= signal.shape[0]/sr 
		temp_results={"file_name":row_data["file_name"].split('/')[-1], "length": length, "sample rate": sr}    
		all_stats=all_stats.append(temp_results, ignore_index=True)
		
		iterator=iterator+1
		#print(iterator)
		if ((iterator>=sample_size) or (output_number_iterator >=required_output_number)):
		    break
		
		if (mode=="just-calculations"):
			continue
		#choosing the random starting point
		if(length<final_target_length):
			print("lengh less than threshold")
			continue
			
		
		detected_time=row_data["detection_time"]
		start_value=max( 0, (detected_time-(final_target_length/2)))  #clipping to the start
		end_value=(start_value+final_target_length)
		if (end_value>length):
			start_value=end_value-final_target_length


		signal=signal[int(start_value*sr):int(end_value*sr)]

		new_length=len(signal)/sr
		
		plot_spectrogram(signal,rate=sr, mode="simple", target_length=new_length).savefig(output_directory+'/batsnet_train/1/'+save_to+'.png')
		#plot_spectrogram(signal,rate=sr, mode="simple", target_length=length).savefig(output_directory+'/spectrograms/original/'+save_to+'.png')
		plt.close()


		random_augmentation=random.randint(0, 1)
		if(random_augmentation==0):
			plot_spectrogram(signal,rate=sr, mode="timeshift", target_length=new_length).savefig(output_directory+'/augmented/'+save_to+'.png')
			plt.close()


		else:
			plot_spectrogram(signal,rate=sr, mode="speedtune",target_length=new_length).savefig(output_directory+'/augmented/'+save_to+'.png')
			plt.close()


		output_number_iterator=output_number_iterator+1

		#break
	except:
		continue
	
#Template to save the stats
#all_stats.to_csv("/media/rabi/Data/ThesisData/audio data analysis/audio-clustering/plots_26april/all_stats.csv")

print("Finished")
