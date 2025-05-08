import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
from pathlib import Path

_FILE_DIR = Path(__file__).resolve().parent#obtain directory of this file
_PROJ_DIR = _FILE_DIR.parent#obtain main project directory
_RESULT_DIR = _PROJ_DIR / "result"#obtain results directory

def visualizeFeature(name: str, melanomaClass: pd.DataFrame, nonMelanomaClass: pd.DataFrame, KDE = False):
    plt.figure(figsize=(8, 5))

    # Convert inputs to numpy arrays if they aren't already
    melanomaClass = np.array(melanomaClass)
    nonMelanomaClass = np.array(nonMelanomaClass)

    if KDE:
        # Define KDE functions
        kde0 = gaussian_kde(melanomaClass)
        kde1 = gaussian_kde(nonMelanomaClass)

        # Define a common range for the x-axis
        min_val = min(melanomaClass.min(), nonMelanomaClass.min())
        max_val = max(melanomaClass.max(), nonMelanomaClass.max())
        x = np.linspace(min_val, max_val, 500)

        # Plot KDEs
        plt.plot(x, kde0(x), label='Melanoma', color='red', alpha=0.7)
        plt.plot(x, kde1(x), label='Non Melanoma', color='blue', alpha=0.7)

        # Optionally add filled areas
        plt.fill_between(x, kde0(x), alpha=0.2, color='red')
        plt.fill_between(x, kde1(x), alpha=0.2, color='blue')
    
    else:
        histBins = np.arange(0, 1.1, 0.1)  # Bin edges: [0.0, 0.1, ..., 1.0]
        # Plot histograms with transparency (alpha)
        plt.hist(melanomaClass, alpha=0.5, bins=histBins, color='red', label='Melanoma', density=True)
        plt.hist(nonMelanomaClass, alpha=0.5, bins=histBins, color='blue', label='Non Melanoma', density=True)

    plt.title(f'Feature Distribution for {name}')
    plt.xlabel(name)
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #save to png
    plt.savefig(str(_RESULT_DIR / f"feature_distribution_{name}.png"), dpi=300, bbox_inches="tight")
    plt.close()#frees up the memory

def visualizeFeature3d(feature1: str, feature2:str, feature3:str, data:pd.DataFrame):
    fig = px.scatter_3d(data, 
                        x=feature1, 
                        y=feature2, 
                        z=feature3,
                        color="true_melanoma_label",
                        opacity=0.8,
                        title=f"Feature Plot for {feature1}, {feature2}, and {feature3}")

    fig.update_traces(marker=dict(size=4, line=dict(width=0.5, color='black')))
    
    # Save to HTML
    fig.write_html(str(_RESULT_DIR / f"features_{feature1}_{feature2}_{feature3}.html"))

def main():
    featureFile = str(_PROJ_DIR / "data/features.csv")
    df=pd.read_csv(featureFile)
    df=df.drop(axis=1,labels=["img_id", "patient_id", "lesion_id", "pat_les_ID"])#drop unnecessary columns
    #get filter for Melanoma and Non Melanoma
    dfMelanoma = df[(df["true_melanoma_label"] == 1)]
    dfNonMelanoma = df[(df["true_melanoma_label"] == 0)]

    dfMelanoma = dfMelanoma.drop(["true_melanoma_label"], axis=1).reset_index()
    dfNonMelanoma = dfNonMelanoma.drop(["true_melanoma_label"], axis=1).reset_index()
    
    visualizeFeature("fA_score", dfMelanoma["fA_score"], dfNonMelanoma["fA_score"], KDE=True)
    visualizeFeature("fB_score", dfMelanoma["fB_score"], dfNonMelanoma["fB_score"], KDE=True)
    visualizeFeature("fC_score", dfMelanoma["fC_score"], dfNonMelanoma["fC_score"], KDE=False)
    visualizeFeature("fBV_score", dfMelanoma["fBV_score"], dfNonMelanoma["fBV_score"], KDE=True)
    visualizeFeature("fCH_score", dfMelanoma["fCH_score"], dfNonMelanoma["fCH_score"], KDE=False)
    visualizeFeature("fS_score", dfMelanoma["fS_score"], dfNonMelanoma["fS_score"], KDE=False)

    visualizeFeature3d("fA_score", "fBV_score", "fS_score", df)



if __name__ == "__main__":
    main()