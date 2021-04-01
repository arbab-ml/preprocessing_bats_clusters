#plotting for IMSAT
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data_directory="/media/rabi/Data/ThesisData/audio data analysis/audio-clustering/plots_15march_b/spectrograms_normalized/results/"
metadata={
    "imsat" : { "file_name": "stats_imsat_10_clusters.csv",
                "y_axis": "Loss",
                "title": "Learning Curve of IMSAT for clustering Bat Calls"
            },
    "iic" : { "file_name": "stats_iic.csv",
                "y_axis": ["First Head Loss", "Second Head Loss"],
                "title": "Learning Curve of IIC for clustering Bat Calls"
            },
    "scan" : { "file_name": "stats_SCAN.csv",
                "y_axis": "Loss",
                "title": "Learning Curve of SCAN for clustering Bat Calls"
            },
    "deepcluster" : { "file_name": "stats_deepcluster.csv",
                "y_axis": ["ConvNet loss","Clustering loss"],
                "title": "Learning Curve of DeepCluster for clustering Bat Calls"
            },

}


plot_turn="deepcluster"


if plot_turn=="scan":
    meta_selected=metadata[plot_turn]
    read_this=meta_selected["file_name"]
    read_csv=pd.read_csv(data_directory+read_this)
    x1=read_csv["Epoch"]
    y1=read_csv[meta_selected["y_axis"]]
    plt.plot(x1, y1, "-b", label="Loss")
    plt.ylim(min(y1), max(y1))
    plt.xlabel("Number of Epochs")
    plt.ylabel(meta_selected["y_axis"])
    plt.title(meta_selected["title"])
    plt.savefig(data_directory+ meta_selected["file_name"]+'.jpeg')

elif plot_turn=='iic':
    meta_selected=metadata[plot_turn]
    read_this=meta_selected["file_name"]
    read_csv=pd.read_csv(data_directory+read_this)
    x1=read_csv["Epoch"]
    y1=read_csv[meta_selected["y_axis"][0]]
    plt.plot(x1, y1, "-b", label=meta_selected["y_axis"][0])

    y2=read_csv[meta_selected["y_axis"][1]]
    plt.plot(x1, y2, "-g", label=meta_selected["y_axis"][1])

    plt.ylim(min( min(y1), min(y2) ), max( max(y1), max(y2) ))
    plt.xlabel("Number of Epochs")
    plt.ylabel("Total Loss")
    plt.title(meta_selected["title"])
    plt.legend(loc="upper right")
    plt.savefig(data_directory+ meta_selected["file_name"]+'.jpeg')
    
elif plot_turn=='scan':
    meta_selected=metadata[plot_turn]
    read_this=meta_selected["file_name"]
    read_csv=pd.read_csv(data_directory+read_this)
    x1=read_csv["Epoch"]
    y1=read_csv[meta_selected["y_axis"]]
    plt.plot(x1, y1, "-b", label="Loss")
    plt.ylim(min(y1), max(y1))
    plt.xlabel("Number of Epochs")
    plt.ylabel(meta_selected["y_axis"])
    plt.title(meta_selected["title"])
    plt.savefig(data_directory+ meta_selected["file_name"]+'.jpeg')
elif plot_turn=='deepcluster':
    fig, ax1 = plt.subplots()

    meta_selected=metadata[plot_turn]
    read_this=meta_selected["file_name"]
    read_csv=pd.read_csv(data_directory+read_this)
    x1=read_csv["Epochs"]
    y1=read_csv[meta_selected["y_axis"][0]]
    ax1.plot(x1, y1, "-b", label=meta_selected["y_axis"][0])

    y2=read_csv[meta_selected["y_axis"][1]]
    ax2=ax1.twinx()
    plt.plot(x1, y2, "-g", label=meta_selected["y_axis"][1])

    #plt.ylim(min( min(y1), min(y2) ), max( max(y1), max(y2) ))
    ax1.set_xlabel("Number of Epochs")
    ax1.set_ylabel("Clustering Loss")
    ax2.set_ylabel("ConvNet Loss")
    ax1.set_title(meta_selected["title"])

    ax1.legend(loc="upper right")
    ax2.legend(loc="center right")
    plt.savefig(data_directory+ meta_selected["file_name"]+'.jpeg')


# elif plot_turn=="iic":

