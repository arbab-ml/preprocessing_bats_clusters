#plotting for IMSAT
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data_directory="/media/rabi/Data/ThesisData/audio data analysis/specie-clustering/Identified calls/all_specie/spectrograms_vmin_vmax_highpass_balance_top5_2_cropped/results/"
metadata={
    "imsat" : { "file_name": "stats_imsat.csv",
                "y_axis": "Loss",
                "title": "Learning Curve of IMSAT for clustering Bat Calls"
            },
    "iic" : { "file_name": "stats_iic.csv",
                "y_axis": ["First Head Loss", "Second Head Loss"],
                "title": "Learning Curve of IIC for clustering Bat Calls"
            },
    "scan" : { "file_name": "stats_SCAN.csv",
                # "y_axis": "Loss",
                "y_axis": ["Pretext Loss", "SCAN Consistency Loss"],
                "title": "Learning Curve of SCAN for clustering Bat Calls"
            },
    "deepcluster" : { "file_name": "stats_deepcluster.csv",
                "y_axis": "ConvNet loss", #old (both in list)
                "title": "Learning Curve of DeepCluster for clustering Bat Calls"
            },
     "k-Medoid" : { "file_name": "kmedoid_5000.csv",
                "y_axis": ["ORB SIM", "SSIM"], #old (both in list)
                "title": " Sum of distances of samples to their closest cluster center in K-Medoid "
            },   
         "k-Medoid-3axis" : { "file_name": "kmedoid_5000.csv",
                "y_axis": ["ORB SIM", "SSIM", "RMSE"], #old (both in list)
                "title": "Sum of distances of samples to their closest cluster center"
            },
         "specie" : { "file_name": "stats_commulative.csv",
                "y_axis": ["IMSAT Loss"	,"IIC (Head A) Loss" ,	"DeepCluster ConvNet Loss","SCAN Consistency Loss"],
                "title": "Learning Curve of Different Algorithms for Specie Clustering "
            },
}


plot_turn="specie"


if plot_turn=="scan":
    # meta_selected=metadata[plot_turn]
    # read_this=meta_selected["file_name"]
    # read_csv=pd.read_csv(data_directory+read_this)
    # x1=read_csv["Epoch"]
    # y1=read_csv[meta_selected["y_axis"]]
    # plt.plot(x1, y1, "-b", label="Loss")
    # plt.ylim(min(y1), max(y1))
    # plt.xlabel("Number of Epochs")
    # plt.ylabel(meta_selected["y_axis"])
    # plt.title(meta_selected["title"])
    # plt.savefig(data_directory+ meta_selected["file_name"]+'.jpeg')

    #2-axis implementation
    fig, ax1 = plt.subplots()

    meta_selected=metadata[plot_turn]
    read_this=meta_selected["file_name"]
    read_csv=pd.read_csv(data_directory+read_this)
    x1=read_csv["Epoch"]
    y1=read_csv[meta_selected["y_axis"][0]]
    ax1.plot(x1, y1, "-b", label=meta_selected["y_axis"][0])

    y2=read_csv[meta_selected["y_axis"][1]]
    ax2=ax1.twinx()
    plt.plot(x1, y2, "-g", label=meta_selected["y_axis"][1])

    #plt.ylim(min( min(y1), min(y2) ), max( max(y1), max(y2) ))
    ax1.set_xlabel("Number of Epochs")
    ax1.set_ylabel(meta_selected["y_axis"][0])
    ax2.set_ylabel(meta_selected["y_axis"][1])
    ax1.set_title(meta_selected["title"])

    ax1.legend(loc="upper right")
    ax2.legend(loc="center right")
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
    
elif plot_turn=='imsat':
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
##############
# elif plot_turn=='deepcluster':
#     fig, ax1 = plt.subplots()

#     meta_selected=metadata[plot_turn]
#     read_this=meta_selected["file_name"]
#     read_csv=pd.read_csv(data_directory+read_this)
#     x1=read_csv["Epochs"]
#     y1=read_csv[meta_selected["y_axis"][0]]
#     ax1.plot(x1, y1, "-b", label=meta_selected["y_axis"][0])

#     y2=read_csv[meta_selected["y_axis"][1]]
#     ax2=ax1.twinx()
#     plt.plot(x1, y2, "-g", label=meta_selected["y_axis"][1])

#     #plt.ylim(min( min(y1), min(y2) ), max( max(y1), max(y2) ))
#     ax1.set_xlabel("Number of Epochs")
#     ax1.set_ylabel("Clustering Loss")
#     ax2.set_ylabel("ConvNet Loss")
#     ax1.set_title(meta_selected["title"])

#     ax1.legend(loc="upper right")
#     ax2.legend(loc="center right")
#     plt.savefig(data_directory+ meta_selected["file_name"]+'.jpeg')
###############
elif plot_turn=='deepcluster':

    meta_selected=metadata[plot_turn]
    read_this=meta_selected["file_name"]
    read_csv=pd.read_csv(data_directory+read_this)
    x1=read_csv["Epochs"]
    y1=read_csv[meta_selected["y_axis"]]
    plt.plot(x1, y1, "-b", label="Loss")
    plt.ylim(min(y1), max(y1))
    plt.xlabel("Number of Epochs")
    plt.ylabel(meta_selected["y_axis"])
    plt.title(meta_selected["title"])
    plt.savefig(data_directory+ meta_selected["file_name"]+'.jpeg')


# elif plot_turn=="iic":
elif plot_turn=='k-Medoid':
    fig, ax1 = plt.subplots()
    meta_selected=metadata[plot_turn]
    read_this=meta_selected["file_name"]
    read_csv=pd.read_csv(data_directory+read_this)
    x1=read_csv["K"]
    y1=read_csv[meta_selected["y_axis"][0]]
    ax1.plot(x1, y1, "-b", label=meta_selected["y_axis"][0])

    y2=read_csv[meta_selected["y_axis"][1]]
    ax2=ax1.twinx()
    plt.plot(x1, y2, "-g", label=meta_selected["y_axis"][1])

    #plt.ylim(min( min(y1), min(y2) ), max( max(y1), max(y2) ))
    ax1.set_xlabel("Number of Target Clusters")
    ax1.set_ylabel(meta_selected["y_axis"][0])
    ax2.set_ylabel(meta_selected["y_axis"][1])
    ax1.set_title(meta_selected["title"])

    ax1.legend(loc="upper right")
    ax2.legend(loc="center right")
    plt.savefig(data_directory+ meta_selected["file_name"]+'.jpeg')


elif plot_turn=='k-Medoid-3axis':

    meta_selected=metadata[plot_turn]
    read_this=meta_selected["file_name"]
    read_csv=pd.read_csv(data_directory+read_this)
    x1=read_csv["K"]
    y1=read_csv[meta_selected["y_axis"][0]]
    y2=read_csv[meta_selected["y_axis"][1]]
    y3=read_csv[meta_selected["y_axis"][2]]

    fig, ax = plt.subplots()
    fig.subplots_adjust(right=0.75)

    twin1 = ax.twinx()
    twin2 = ax.twinx()

    # Offset the right spine of twin2.  The ticks and label have already been
    # placed on the right by twinx above.
    # twin2.spines.right.set_position(("axes", 1.2))
    twin2.spines['right'].set_position(("axes", 1.2))


    p1, = ax.plot(x1,y1, "b-", label=meta_selected["y_axis"][0])
    p2, = twin1.plot(x1, y2, "r-", label=meta_selected["y_axis"][1])
    p3, = twin2.plot(x1, y3, "g-", label=meta_selected["y_axis"][2])
    ax.set_title(meta_selected["title"])
    # ax.set_xlim(0, 2)
    # ax.set_ylim(0, 2)
    # twin1.set_ylim(0, 4)
    # twin2.set_ylim(1, 65)

    ax.set_xlabel("Number of Target Clusters")
    ax.set_ylabel(meta_selected["y_axis"][0])
    twin1.set_ylabel(meta_selected["y_axis"][1])
    twin2.set_ylabel(meta_selected["y_axis"][2])

    ax.yaxis.label.set_color(p1.get_color())
    twin1.yaxis.label.set_color(p2.get_color())
    twin2.yaxis.label.set_color(p3.get_color())

    tkw = dict(size=4, width=1.5)
    ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
    twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    twin2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    ax.tick_params(axis='x', **tkw)

    ax.legend(handles=[p1, p2, p3])
    plt.savefig(data_directory+ meta_selected["file_name"]+'.jpeg')

    # plt.show()


elif plot_turn=='specie':

    px = 1/plt.rcParams['figure.dpi']

    meta_selected=metadata[plot_turn]
    read_this=meta_selected["file_name"]
    read_csv=pd.read_csv(data_directory+read_this).iloc[0:100,]
    x1=read_csv["Epoch"]
    y1=read_csv[meta_selected["y_axis"][0]]
    y2=read_csv[meta_selected["y_axis"][1]]
    y3=read_csv[meta_selected["y_axis"][2]]
    y4=read_csv[meta_selected["y_axis"][3]]


    fig, ax = plt.subplots(figsize=(2*600*px, 600*px))
    fig.subplots_adjust(right=0.75)

    twin1 = ax.twinx()
    twin2 = ax.twinx()
    twin3 = ax.twinx()
        
    # Offset the right spine of twin2.  The ticks and label have already been
    # placed on the right by twinx above.
    # twin2.spines.right.set_position(("axes", 1.2))
    twin2.spines['right'].set_position(("axes", 1.1))
    twin3.spines['right'].set_position(("axes", 1.2))

    p1, = ax.plot(x1,y1, "b-", label=meta_selected["y_axis"][0])
    p2, = twin1.plot(x1, y2, "r-", label=meta_selected["y_axis"][1])
    p3, = twin2.plot(x1, y3, "g-", label=meta_selected["y_axis"][2])
    p4, = twin3.plot(x1, y4, "y-", label=meta_selected["y_axis"][3])
    
    ax.set_title(meta_selected["title"])
    # ax.set_xlim(0, 2)
    # ax.set_ylim(0, 2)
    # twin1.set_ylim(0, 4)
    # twin2.set_ylim(1, 65)

    ax.set_xlabel("Number of Epochs")
    ax.set_ylabel(meta_selected["y_axis"][0])
    twin1.set_ylabel(meta_selected["y_axis"][1])
    twin2.set_ylabel(meta_selected["y_axis"][2])
    twin3.set_ylabel(meta_selected["y_axis"][3])


    ax.yaxis.label.set_color(p1.get_color())
    twin1.yaxis.label.set_color(p2.get_color())
    twin2.yaxis.label.set_color(p3.get_color())
    twin3.yaxis.label.set_color(p4.get_color())

    tkw = dict(size=4, width=1.5)
    ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
    twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    twin2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    twin3.tick_params(axis='y', colors=p4.get_color(), **tkw)
    
    ax.tick_params(axis='x', **tkw)

    ax.legend(handles=[p1, p2, p3,p4])
    plt.savefig(data_directory+ meta_selected["file_name"]+'.jpeg')

