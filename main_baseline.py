import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from util.feature_extraction import extract
from util.classifier import split_data, runClassifier

#RUN the baseline model 

def main(csvPath:str, savePath:str, testImgDir:str, testMaskDir:str, testMetadataPath:str) -> None:
    #obtain training data from our extracted features csv
    datasetDf = pd.read_csv(csvPath).drop(axis=1, labels=["patient_id", "lesion_id"])#drop unnecessary columns
    datasetY = datasetDf['true_melanoma_label']#obtain melanoma binary label-column as y-data
    datasetX = datasetDf.drop("true_melanoma_label", axis=1)#drop melanoma label -> obtain X-data
    xTrain, yTrain, xTest, yTest = split_data(datasetX, datasetY, "pat_les_ID")#split grouping by lesion
    #xTrain = xTrain.drop(axis=1, labels=["pat_les_ID", "img_id"])#get rid of pat_les_id & img_id in training x data
    #xTest = xTest.drop(["pat_les_ID"], axis=1)#get rid of pat_les_id in test x data (BUT keeps img_id in test for later result output)

    #obtain test data if testImgDir is provided
    if testImgDir is not None:
        #extract the features from the test images and return them without saving to file
        testData = extract(testImgDir, testMaskDir, testMetadataPath, feature_dir=None, base_model=True)#could also get rid of base_model=True
        testData.drop(axis=1, labels=["img_id", "patient_id", "lesion_id"])#drop unnecessary columns
        if testMetadataPath != None:
            #obtain true labels for testData if available
            yTest = testData["true_melanoma_label"]
            xTest = testData.drop("true_melanoma_label", axis=1)
        else:#set true labels to None if no true labels are available
            yTest = None
            xTest = testData
    

    #NOTE: HERE is the place where we can input our code for the base method
    #      (i.e. creating the classifier with the correct hyperparameters etc...)
    #      + select the correct features

    #select features for baseline method
    xTrain = xTrain[["fA_score", "fB_score", "fC_score"]]
    xTest = xTest[["img_id", "fA_score", "fB_score", "fC_score"]]#keep img_id for test data for later result output

    #create classifier for baseline method
    classifier = RandomForestClassifier(class_weight="balanced",max_depth=4)

    #run classifier and output test results if available
    result = runClassifier(classifier, "base", 0.5, xTrain, yTrain, xTest, yTest)

    #NOTE: END of code for base method
    
    #save result:
    result.to_csv(savePath, index=False)
    print(f"Result saved to: {savePath}")


if __name__ == "__main__":
    testImgDir = None#path to directory with test lesion images (if None -> fallback to use test data from our dataset)
    testMaskDir = None#path to mask directory for test images (optional, if None, assumed to be equal to testImgDir)
    metadataPath = None#path to metadata.csv file for the test images (optional, needed to evaluate performance based on true label)

    trainCSV = "data/features.csv"#path to trainingData csv (features extracted by our method)
    resultCSV = "result/result_baseline.csv"#path where result csv will be saved
    main(trainCSV, resultCSV, testImgDir, testMaskDir, metadataPath)