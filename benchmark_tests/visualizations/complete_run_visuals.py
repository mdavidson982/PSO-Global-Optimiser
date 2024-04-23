import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def make_mpso_vs_mpsoccd_graph(path, fnname):
    save_path = os.path.join(path, "Run_figs")
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    folders: list[str] = os.listdir(path)
    if "tests" in folders:
        path = os.path.join(path, "tests")
        folders: list[str] = os.listdir(path)

    folders = [folder for folder in folders if folder.endswith(f"quality-{fnname}")]
    if len(folders) != 2:
        raise Exception(f"Not a correct amount of folders, have {folders}")
    
    mpsoccd_folder = os.path.join(path, [folder for folder in folders if "ccd" in folder][0])
    mpso_folder = os.path.join(path, [folder for folder in folders if "ccd" not in folder][0])

    mpsoccd_data = pd.read_csv(os.path.join(mpsoccd_folder, "MPSORuns.csv"))
    mpso_data = pd.read_csv(os.path.join(mpso_folder, "MPSORuns.csv"))
    
    best_mpsoccd_replicate = mpsoccd_data.loc[mpsoccd_data["g_best_value_ccd"].argmin()]["replicate_number"]
    best_mpso_replicate = mpso_data.loc[mpso_data["g_best_value"].argmin()]["replicate_number"]

    mpsoccd_best_replicate_folder = os.path.join(mpsoccd_folder, "MPSORuns", f"MPSO-Iteration-{best_mpsoccd_replicate}")
    mpso_best_replicate_folder = os.path.join(mpso_folder, "MPSORuns", f"MPSO-Iteration-{best_mpso_replicate}")

    best_mpsoccd_data = pd.read_csv(os.path.join(mpsoccd_best_replicate_folder, "MPSOData.csv"))
    best_mpso_data = pd.read_csv(os.path.join(mpso_best_replicate_folder, "MPSOData.csv"))

    title = f"MPSO vs MPSO CCD Quality over cycles-{fnname}"
    fig = plt.figure()
    plt.plot(
        np.array(best_mpsoccd_data["mpso_iteration"]), 
        np.array(best_mpsoccd_data["g_best_value_ccd"]), 
        label = "MPSOCCD quality", 
        color = "green",
    )
    
    plt.plot(
        np.array(best_mpso_data["mpso_iteration"]), 
        np.array(best_mpso_data["g_best_value"]), 
        label = "MPSO quality", 
        color = "blue",
        linestyle = "dotted"
    )

    plt.title(title)
    plt.xlabel("MPSO/MPSOCCD Cycles")
    plt.ylabel("Quality")
    plt.legend()
    fig.savefig(os.path.join(save_path, title))
    plt.close(fig)