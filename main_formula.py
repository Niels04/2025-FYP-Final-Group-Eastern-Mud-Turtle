from pathlib import Path
from util.feature_extraction import extract
from util.classifier import Formula

_PROJ_DIR = Path(__file__).resolve().parent   # obtain project directory
_DATA_DIR = _PROJ_DIR / "data"                # obtain data directory
_RESULT_DIR = _PROJ_DIR / "result"            # obtain results directory


def main(savePath:str, testImgDir:str, testMaskDir:str, testMetadataPath:str) -> None:

    #vobtain external test data if testImgDir is provided
    if testImgDir is not None:

        # extract the features from the external test images and return them without saving to file
        testData = extract(testImgDir, testMaskDir, testMetadataPath, feature_dir= None, formula_features= 'only')
        testData.drop(axis=1, labels=["patient_id", "lesion_id"]) # drop unnecessary columns BUT keep img_id for later result output
        
        if testMetadataPath is not None:
            
            # obtain true labels for testData if available
            y = testData["true_melanoma_label"]
            x = testData.drop("true_melanoma_label", axis=1)
        
        else: # set true labels to None if no true labels are available
            y = None
            x = testData

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
    testImgDir = None # path to directory with external test lesion images (if None -> fallback to use test data from our dataset)
    testMaskDir = None # path to mask directory for external test images (optional, if None, assumed to be equal to testImgDir)
    metadataPath = None # path to metadata.csv file for the external test images (optional, needed to evaluate performance based on true label)

    trainCSV = str(_DATA_DIR / "features.csv") # path to trainingData csv (features extracted by our method)
    resultCSV = str(_RESULT_DIR / "result_extended.csv") # path where result csv will be saved
    main(resultCSV, testImgDir, testMaskDir, metadataPath)