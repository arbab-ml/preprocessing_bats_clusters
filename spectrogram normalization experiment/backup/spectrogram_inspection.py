import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
from pathlib import Path
#Ref: https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53
def plot_spectrogram(signals, labels_flag=False, rate=44100,mode="simple", target_length=3):
	bottom_value=20000
	top_value=92000

	fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False,
								 sharey=True, figsize=(5,5))
	fig.suptitle('Spectrogram', size=16)
	#axes.set_title(list(fbank.keys())[i])
	hop_length_value=32
	if (mode=="simple"):
		spec = np.abs(librosa.stft(signals, hop_length=hop_length_value, n_fft=5120))
		spec = librosa.amplitude_to_db(spec, ref=np.max)
		axes=librosa.display.specshow(spec, sr=rate,hop_length=hop_length_value, x_axis='time', y_axis='linear') #when computing an STFT, you can pass that same parameter to specshow. 
																					#This ensures that axis scales (e.g. time or frequency) are computed correctly.
		ax2 = fig.gca() #getting this to set the range
		print("Before adjusting axes, the yaxis range was: ", ax2.get_ylim())
		ax2.set_ylim(bottom=bottom_value, top=top_value) # Setting frequency betweei 20KHz and 92KHz
		return fig


#########
input_directory = r'/home/rabi/Documents/Thesis/spectrogram normalization experiment'
#output_directory=r'/home/rabi/Documents/Thesis/audio data analysis/audio-clustering/all plots'
output_directory=r'/home/rabi/Documents/Thesis/spectrogram normalization experiment'
bat_calls_data_dir=r'//home/rabi/Documents/Thesis/spectrogram normalization experiment/results.csv'
final_target_length=0.5 #1 second

bat_calls_data=pd.read_csv(bat_calls_data_dir)
bat_calls_data_processed=bat_calls_data.iloc[bat_calls_data.groupby('file_name')['detection_prob'].agg(pd.Series.idxmax)]
bat_calls_data_processed["just_file_name"]= bat_calls_data_processed.iloc[:,0].apply(lambda x: x.split('/')[-1])
bat_calls_data_processed=bat_calls_data_processed.set_index("just_file_name")


for path in (Path(input_directory).rglob('*.wav')):
	signal, sr = librosa.load(path,sr=None)  
	just_name=str(path).split('/')[-1]
	detected_time=bat_calls_data_processed.loc[just_name]['detection_time']

	start_value=(detected_time-(final_target_length/2))
	end_value=(detected_time+(final_target_length/2))

	signal=signal[int(start_value*sr):int(end_value*sr)]
	#Now saving the statistics (sr and length)
	#filename_short=save_to.split('/')[-1]
	#Now, storing the plots       
	# plot_signal(signal,labels_flag=True).savefig(output_directory+'/signals/'+save_to+'.png')
	# plot_fft(calc_fft(signal, sr),labels_flag=True).savefig(output_directory+'/ffts/'+save_to+'.png')
	# plot_spectrogram(signal).savefig(output_directory+'/spectrograms/'+save_to+'.png')
	#plot_mel_spectrogram(signal, rate=sr).savefig(output_directory+'/mel spectrograms/'+save_to+'.png')
	#Making the plots, 3 modes are available = simple, speedtune, timeshift
	new_length=len(signal)/sr
	plot_spectrogram(signal,rate=sr, mode="simple", target_length=new_length).savefig(output_directory+'/'+just_name+'.png')