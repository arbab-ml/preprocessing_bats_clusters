# Scripts for Preprocessing and other utility functions 
These scripts are used for preprocessing, evaluation, and plots generation. The purpose of each script is given below:

#### `v4 precomputation generic utility.py`
Generates the equivalent file to batdetect for non-batdetect usage. It could be used on any dataset (bat/non-bat).  It takes as input:
* Length of Audio files
* Input directory containing raw audio files(named according to their labels)
* Output directory for resulting file
 
#### `v4 precomputation generic.py`
Generates Spectrograms and augmentations for both the labelled and unlabellled data  It takes as input:
* target sample rate
* Input directory containing raw audio files
* The output file from batdetect (which contain the location of individual bat call in each audio recording). Or the output from `v4 precomputation generic utility.py` for non-batdetect usage. 
* Output directory for saving the spectrograms


#### `resize_images_spectrograms.py`
Removes the axis labels, axis ticks, and titles from all the images (like `crop_images_spectrograms.py` ) and also resizes the spectrograms to 128x128 size. It takes the directory as input.



#### `calculate_stats.py`
Calculates the sample rate and length of all the audio files in a directory. It takes the directory as input. 

#### `NMI_calculation_all.py`
Calculates `normalized_mutual_info_score,mutual_info_score,
 adjusted_mutual_info_score, adjusted_rand_score, completeness_score, fowlkes_mallows_score, homogeneity_score, v_measure_score` between the output of clustering algorithms. 
The input directory and output files names are given as input. 

#### `conf_matrix_all.py.py`
Calculates Cross Tabulation between the output of different algorithms. The input directory and output files names are given as input.  

#### `crop_images_spectrograms.py`
Remove axis labels, axis ticks, and titles from all the images in a directory. It takes the directory as input.

#### `draw_histogram.py`
Draws the historgram of Length of Audio Files (5 Bins). It takes as input the `csv` file containing statistic of all individual files. 

#### `filename_to_image_rename.py`
It converts the raw `csv` output of an algorithm to images, which are divided in their respective cluster's folder. It takes as input the 
* Directory containing all the spectrograms
* `csv` output file of any algorithm.
* Output diretory path (where the results should be stored)

#### `labelled_data_evaluation.py`
Calculates the evaluation metrics of results form labeled data. It takes as input the resulting `csv` file after training a model on labeled data.

#### `loss_plots.py`
Plots the learning curve of individual and multiple algorithms. The input directory and loss values from each iteration (for an algorithm) are given as input.  

####  `bat_detect/bat_eval/run_detector.py`

Python code for the detection of bat echolocation calls in full spectrum audio recordings. This code recreate the results from the paper [Bat Detective - Deep Learning Tools for Bat Acoustic Signal Detection](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005995). You will also find some additional information and data on our [project page](http://visual.cs.ucl.ac.uk/pubs/batDetective). For original implementation please refer to [This link](https://github.com/macaodha/batdetect)

`bat_eval` contains lightweight python scripts that load a pretrained model and run the detector on a directory of audio files. No GPU is required for this step.  

It takes as input (from line `90` ownwards)
* Input directory containing raw audio files
* Output directory where to save the results
* Bat call detection confidence threshold 

**Deprecated Files**: `v1 precomputation.py`, `v2 precomputation_kaleidoscope.py` `v3_specie precomputation all from batdetect.py` `v3 precomputation all from batdetect.py`
