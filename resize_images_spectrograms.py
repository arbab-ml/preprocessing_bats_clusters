from PIL import Image
import os.path, sys
from pathlib import Path
from tqdm import tqdm
# /media/rabi/Data/ThesisData/audio data analysis/audio-clustering/plots_26april/spectrograms_normalized
input_path = "/media/rabi/Data/ThesisData/audio data analysis/audio-clustering/plots_26april/spectrograms_normalized_croped_128/results/interpretation with coding scheme/representative samples of each algorithm"
output_path= "/media/rabi/Data/ThesisData/audio data analysis/audio-clustering/plots_26april/spectrograms_normalized_croped_128/results/interpretation with coding scheme/representative samples of each algorithm/cropped spectrograms"
dirs = Path(input_path).rglob('*.png')
newsize=(128,128)
def crop():
    for item in tqdm(dirs):
        fullpath = item       #corrected
        im = Image.open(fullpath)
        imCrop = im.crop((62, 58, 62+390, 58+390)).resize(newsize) #corrected
        if ( str(fullpath).split('/')[-3] =='batsnet_train'):
            new_file_name=output_path +'/'+ str(fullpath).split('/')[-3] +   '/'+             str(fullpath).split('/')[-2]+'/'+str(fullpath).split('/')[-1]
        else:
            # continue
            new_file_name=output_path +'/'+ str(fullpath).split('/')[-2]+'/'+str(fullpath).split('/')[-1]
        imCrop.save(new_file_name , "png", quality=100)

crop()