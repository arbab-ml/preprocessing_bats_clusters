from PIL import Image
import os.path, sys
from pathlib import Path

input_path = "/media/rabi/Data/ThesisData/audio data analysis/audio-clustering/plots_15march_b/spectrograms_normalized/representative samples of each algorithm_8thApril"
dirs = Path(input_path).rglob('*.png')

def crop():
    for item in dirs:
        fullpath = item       #corrected
        im = Image.open(fullpath)
        imCrop = im.crop((62, 58, 62+390, 58+390)) #corrected
        new_file_name=input_path +'/cropped spectrograms/'+ str(fullpath).split('/')[-2]+'/'+str(fullpath).split('/')[-1]
        imCrop.save(new_file_name , "png", quality=100)

crop()