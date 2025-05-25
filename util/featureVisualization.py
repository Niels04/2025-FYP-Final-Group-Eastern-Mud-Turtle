import numpy as np
import pandas as pd
from pathlib import Path

from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

_FILE_DIR = Path(__file__).resolve().parent#obtain directory of this file
_PROJ_DIR = _FILE_DIR.parent#obtain main project directory
_PLOT_DIR = _PROJ_DIR / "result/featurePlots"#obtain featurePlots directory

def visualizeFeature(feature: str, data: pd.DataFrame, KDE = False) -> None:
    """Outputs a histogram or density plot for the given feature.\n
    Overlays the distributions for both melanoma and non-melanoma.\n

    :param feature: name of the feature
    :param data: dataFrame for the feature data
    :param KDE: Whether to create a histogram or a smoothed-out density plot with Kernel Density Estimation
    :return None:"""

    #first filter in melanoma and non-melanoma points
    melanoma = (data[data["true_melanoma_label"] == 1])[feature]
    nonMelanoma = (data[data["true_melanoma_label"] == 0])[feature]

    plt.figure(figsize=(8, 5))

    if KDE:
        #create KDE functions
        kde0 = gaussian_kde(melanoma)
        kde1 = gaussian_kde(nonMelanoma)

        #calculate common range for x-axis
        min_val = min(melanoma.min(), nonMelanoma.min())
        max_val = max(melanoma.max(), nonMelanoma.max())
        x = np.linspace(min_val, max_val, 500)

        ##plot KDE
        plt.plot(x, kde0(x), label='Melanoma', color='red', alpha=0.7)
        plt.plot(x, kde1(x), label='Non Melanoma', color='blue', alpha=0.7)

        #fill areas under the curves
        plt.fill_between(x, kde0(x), alpha=0.2, color='red')
        plt.fill_between(x, kde1(x), alpha=0.2, color='blue')
    
    else:#make histogram
        histBins = np.arange(0, 1.1, 0.1)#bin edges at [0.0, 0.1, ..., 1.0]
        
        plt.hist(melanoma, alpha=0.5, bins=histBins, color='red', label='Melanoma', density=True)
        plt.hist(nonMelanoma, alpha=0.5, bins=histBins, color='blue', label='Non Melanoma', density=True)

    plt.title(f'Feature Distribution for {feature}')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #save to svg
    plt.savefig(str(_PLOT_DIR / f"feature_distribution_{feature}.svg"), dpi=300, bbox_inches="tight")
    plt.close()#frees up the memory

def visualizeFeature2d(feature1: str, feature2:str, data:pd.DataFrame) -> None:
    """Given 2 names of feature columns and the data frame containing these freatures,\n
    ouputs a scatterplot in the results directory that visualizes these 2 features.\n
    Points are colored based on their class.\n
    
    :param feature1: name of the column for 1st feature
    :param feature2: name of the column for 2nd feature
    :param data: dataFrame for the features
    :return None:"""
    
    #first filter in melanoma and non-melanoma points
    melanomaDf = data[data["true_melanoma_label"] == 1]
    nonMelanomaDf = data[data["true_melanoma_label"] == 0]

    #scatter both melanoma and non melanoma points
    plt.figure(figsize=(8, 6))
    plt.scatter(melanomaDf[feature1], melanomaDf[feature2], 
                    c="red", label="Melanoma", edgecolor='k', alpha=0.5)
    plt.scatter(nonMelanomaDf[feature1], nonMelanomaDf[feature2], 
                    c="blue", label="Non Melanoma", edgecolor='k', alpha=0.5)

    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title(f"Feature Plot for {feature1} and {feature2}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    #save as svg
    plt.savefig(str(_PLOT_DIR / f"features_{feature1}_{feature2}.svg"), dpi=300, bbox_inches="tight")
    plt.close()#free memory

def visualizeFeature3d(feature1: str, feature2:str, feature3:str, data:pd.DataFrame) -> None:
    """Given 3 names of feature columns and the data frame containing these freatures,\n
    ouputs an interactive .html 3d scatterplot in the results directory that visualizes these 3 features.\n
    Points are colored based on their class.
    
    :param feature1: name of the column for 1st feature
    :param feature2: name of the column for 2nd feature
    :param feature3: name of the column for 3rd feature
    :param data: dataFrame for the features
    :return None:"""
    fig = px.scatter_3d(data, 
                        x=feature1, 
                        y=feature2, 
                        z=feature3,
                        color="true_melanoma_label",
                        opacity=0.8,
                        title=f"Feature Plot for {feature1}, {feature2}, and {feature3}")

    fig.update_traces(marker=dict(size=4, line=dict(width=0.5, color='black')))
    
    #save
    fig.write_html(str(_PLOT_DIR / f"features_{feature1}_{feature2}_{feature3}.html"))

def main():
    #first, read the features 
    featureFile = str(_PROJ_DIR / "data/features.csv")
    df=pd.read_csv(featureFile)
    df=df.drop(axis=1,labels=["img_id", "patient_id", "lesion_id", "pat_les_ID"])#drop unnecessary columns (keep true_melanoma_label for visualization purposes)
    
    # NOTE: Playground for visualizing and (hopefully) understanding featuers:

    visualizeFeature("fA_score", df, KDE=True)
    visualizeFeature("fB_score", df, KDE=True)
    visualizeFeature("fC_score", df, KDE=False)
    visualizeFeature("fBV_score", df, KDE=True)
    visualizeFeature("fCH_score", df, KDE=False)
    visualizeFeature("fS_score", df, KDE=False)

    visualizeFeature3d("fA_score", "fBV_score", "fS_score", df)
    visualizeFeature3d("fA_score", "fBV_score", "fC_score", df)
    visualizeFeature2d("fA_score", "fBV_score", df)



if __name__ == "__main__":
    main()