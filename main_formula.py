from codecarbon import track_emissions
from pathlib import Path
from util.feature_extraction import extract
from util.classifierOpenQuestion import Formula
import pandas as pd

_PROJ_DIR = Path(__file__).resolve().parent   # obtain project directory
_DATA_DIR = _PROJ_DIR / "data"                # obtain data directory
_RESULT_DIR = _PROJ_DIR / "result"            # obtain results directory

@track_emissions(country_iso_code = "DNK")
def main(savePath:str, MetadataPath:str) -> None:

    md = pd.read_csv(MetadataPath)
            
    # obtain true labels for testData if available
    y = md["true_melanoma_label"]
    x = md.drop("true_melanoma_label", axis=1)


    # select features for the method
    x = x[["img_id", "A_val", "B_val", "C_val", "D_val"]] # keep img_id for test data for later result output

    # create classifier for the method
    classifier = Formula()

    # run classifier and output test results if available
    result = classifier.runFormulaClassifier(x, y)

    #NOTE: END of code for extended method
    
    # save result:
    result.to_csv(savePath, index=False)
    print(f"Result saved to: {savePath}")


if __name__ == "__main__":
    metadataPath = str(_DATA_DIR / "features.csv") # path to metadata.csv file for the external test images (optional, needed to evaluate performance based on true label)
    resultCSV = str(_RESULT_DIR / "result_formula.csv") # path where result csv will be saved
    main(resultCSV, metadataPath)