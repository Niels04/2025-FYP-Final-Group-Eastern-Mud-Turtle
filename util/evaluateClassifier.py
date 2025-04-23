import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

class Performance:
    """Class to store performance metrics
    of a method on either test or validation/test data.
    """
    def __init__(self, ACCs: pd.DataFrame, RECs: pd.DataFrame, AUCs: pd.DataFrame, meanAcc:float, varAcc:float, meanRecall:float, varRecall:float, meanAUC:float, varAUC:float):
        self.meanAcc = meanAcc
        self.varAcc = varAcc
        self.meanRecall = meanRecall
        self.varRecall = varRecall
        self.meanAUC = meanAUC
        self.varAUC = varAUC
        #necessary to save actual performances for later boxplot generation
        self.ACCs = ACCs
        self.RECs = RECs
        self.AUCs = AUCs
    def __str__(self):
        return (
            f"\tMean Accuracy: {self.meanAcc:.4f}\n"
            f"\tVariance Accuracy: {self.varAcc:.4f}\n"
            f"\tMean Recall: {self.meanRecall:.4f}\n"
            f"\tVariance Recall: {self.varRecall:.4f}\n"
            f"\tMean AUC: {self.meanAUC:.4f}\n"
            f"\tVariance AUC: {self.varAUC:.4f}"
        )

class MethodPerformance:
    """Class to store performance metrics
    for a particular method on its training
    and validation/test data."""
    def __init__(self, trainPerf: Performance, valPerf: Performance):
        self.trainPerformance = trainPerf
        self.validationPerformance = valPerf

class Evaluator:
    def __init__(self):
        self.performances = {}
    
    #NOTE: This is for our own iterative process of finding the best classifier/trainSize/hyperparameters like tree depth, etc...
    #      For the final evaluation we could slightly re-write this method to measure prediction accuracy on actual test data
    #      (not validation data like its using right now)
    #      BUT I believe we shouldn't iterate & change our model after running it on the actual test data
    #      (would be Overfitting by Observer as described in the lecture)
    #      so I think it's super important that we keep "random_state=42" in the split_data the whole time

    #      To be added:
    #       - method that evaluates performance with test data (only use at end) -> could use for hypothesis testing
    #       - 
    def evalClassifier(self, classifier, name:str, xTrain:pd.DataFrame, yTrain:pd.DataFrame, patientGroups:pd.DataFrame, threshold:float, nShuffles = 20, validationSize=0.2, saveCurveROC = False):
        """Given the classifier, computes AUC(accuracy) and
        recall (TP/(TP+FN)) over given number of grouped
        shuffles of the given training data, grouped by the
        given column.\n
        Stores results in Evaluator class (can be printed with
        \"printPerformances()\").

        :param classifer: The untrained classifier to be evaluated
        :param name: Classifier name string
        :param xTrain: The x-columns of the training/working data
        :param yTrain: The y-column of the training/working data
        :param patientGroups: Column of training/working data to group by
        :param threshold: Decision Threshold from 0 to 1 (0.4 means that everything with probability of melanoma > 0.4 will be classified as melanoma)
        :param nShuffles: Number of different shuffled folds to perform, default=20
        :param validationSize: Proportion of training/working data to be used as validation Data, default=0.2
        :return None:"""
        ACCsTrain = np.zeros(nShuffles)#array to store accuracy scores for training data
        RECsTrain = np.zeros(nShuffles)#array to store recall scores for training data
        AUCsTrain = np.zeros(nShuffles)#array to store AUC scores (area under ROC curve) for training data
        ACCsVal = np.zeros(nShuffles)#array to store accuracy scores for validation data
        RECsVal = np.zeros(nShuffles)#array to store recall scores for validation data
        AUCsVal = np.zeros(nShuffles)#array to store AUC scores (area under ROC curve) for validation data
        
        #only for generating ROC curve over all shuffles (for validation data)
        allYLabels = []
        allYPredictionProbs = []

        #iterate through all splits, train the model and evaluate performance
        gss=GroupShuffleSplit(n_splits=nShuffles, test_size=validationSize)#not using random_state here for complete randomness
        for i, (trainIdx, valIdx) in enumerate(gss.split(xTrain, yTrain, patientGroups)):
            #FIT CLASSIFIER ON CURRENT SPLIT
            classifier.fit(xTrain.iloc[trainIdx], yTrain.iloc[trainIdx])

            #TEST CLASSIFIER ON CURRENT SPLIT

            #test on training data for current split---------------
            yProbs = classifier.predict_proba(xTrain.iloc[trainIdx])[:, 1]#predict melanoma probability for training data
            #calculate AUC for current shuffle using the prediction probabilities
            AUCsTrain[i] = roc_auc_score(yTrain.iloc[trainIdx], yProbs)
            #turn predicted probabilities into binary predictions using the given decision threshold
            yPred = (yProbs >= threshold).astype(int)
            #compute accuracy and recall for the given decision threshold
            ACCsTrain[i] = accuracy_score(yTrain.iloc[trainIdx], yPred)
            RECsTrain[i] = recall_score(yTrain.iloc[trainIdx], yPred)
            
            #test on validation data for current split--------------
            yProbs = classifier.predict_proba(xTrain.iloc[valIdx])[:, 1]#predict melanoma probability for evaluation data
            if saveCurveROC:
                allYPredictionProbs.extend(yProbs)#save prediction probabilities for current shuffle for combined ROC curve computation
                allYLabels.extend(yTrain.iloc[valIdx])#save true labels for current shuffle for combined ROC curve computation
            #calculate AUC for current shuffle using the prediction probabilities
            AUCsVal[i] = roc_auc_score(yTrain.iloc[valIdx], yProbs)
            #turn predicted probabilities into binary predictions using the given decision threshold
            yPred = (yProbs >= threshold).astype(int)
            #compute accuracy and recall for the given decision threshold
            ACCsVal[i] = accuracy_score(yTrain.iloc[valIdx], yPred)
            RECsVal[i] = recall_score(yTrain.iloc[valIdx], yPred)

        trainPerformance = Performance(ACCsTrain, RECsTrain, AUCsTrain, np.mean(ACCsTrain), np.var(ACCsTrain), np.mean(RECsTrain), np.var(RECsTrain), np.mean(AUCsTrain), np.var(AUCsTrain))
        validationPerformance = Performance(ACCsVal, RECsVal, AUCsVal, np.mean(ACCsVal), np.var(ACCsVal), np.mean(RECsVal), np.var(RECsVal), np.mean(AUCsVal), np.var(AUCsVal))
        self.performances[name] = MethodPerformance(trainPerformance, validationPerformance)#store performance for current method in dict

        if saveCurveROC:#compute a combined ROC curve that takes into account predictions over all shuffles and save it to .png
            self.makeGraphROC(name, allYLabels, allYPredictionProbs, dataType="validation")
        #NOTE: "Combined" means that the ROC curve is computed based on the predicted probabilities over all of the random grouped
        #      shuffles FOR THE VALIDATION DATA
    
    def printPerformances(self) -> None:
        for name, perf in self.performances.items():
            print(f"Performance for method \"{name}\"")
            print("On training data:")
            print(perf.trainPerformance)
            print("On validation data:")
            print(perf.validationPerformance)
            print("\n\n")

    def makeGraphROC(self, name:str, yLabels: pd.DataFrame, yPredictedProbs: pd.DataFrame, dataType:str) -> None:
        """Creates an ROC curve given a column of true yLabels and predicted yProbabilities.\n
        Saves the plotted curve as .png \"/result/\"with the provided method name as a suffix.\n
        :param name: name of the method for which the curve is generated
        :param yLabels: true yLabels
        :param yPredictedProbs: probabilities for label=1 generated by method
        :param dataType: specify if input data was training/validation/test data
        :return None:"""

        fPosRate, tPosRate, _ = roc_curve(yLabels, yPredictedProbs)
        AUCscore = roc_auc_score(yLabels, yPredictedProbs)

        plt.figure(figsize=(8, 6))
        plt.plot(fPosRate, tPosRate, label=f'AUC = {AUCscore:.2f}', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')#display random guess middle line

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Combined ROC Curve for method \"{name}\" on {dataType} data')
        plt.legend(loc='lower right')
        plt.grid(True)

        #save to png
        plt.savefig(f"../result/roc_curve_{name}.png", dpi=300, bbox_inches='tight')
        plt.close()#frees up the memory

    def makeBoxplot(self, metric:str) -> None:
        """Generates a boxplot that compares the performances of all different
        methods stored inside evaluator class in a boxplot and saves it as .png in \"/result/\"\n
        :param metric: Performance metric to be compared, one of either: \"recall\", \"accuracy\" or \"AUC\"
        :return None:"""
        #obtain training and validation performances for all methods in a flattened list
        trainPerfs = []
        if metric == "recall":
            trainPerfs = [val for perf in self.performances.values() for val in (perf.trainPerformance.RECs, perf.validationPerformance.RECs)]
        elif metric == "accuracy":
            trainPerfs = [val for perf in self.performances.values() for val in (perf.trainPerformance.ACCs, perf.validationPerformance.ACCs)]
        elif metric == "AUC":
            trainPerfs = [val for perf in self.performances.values() for val in (perf.trainPerformance.AUCs, perf.validationPerformance.AUCs)]
        else:
            raise Exception("\"metric\" must be either \"recall\", \"accuracy\" or \"AUC\".")
        #obtain names of the corresponding methods
        methodNames = list(self.performances.keys())
        methodNames = [f"{name} ({suffix})" for name in methodNames for suffix in ("train", "val")]
        #add boxplot code here!
        #create boxplot
        fig, axes = plt.subplots(figsize=(10, 6))
        box = axes.boxplot(trainPerfs)
        axes.set_xticklabels(methodNames, rotation=45, ha='right')
        axes.set_title(f"Method {metric} Comparison")
        axes.set_ylabel(metric)
        axes.set_xlabel("Method")
        axes.grid(True, linestyle='--', alpha=0.7)

        #save plot
        plt.tight_layout()
        plt.savefig(f"../result/classifier_performance_boxplot_{metric}.png", dpi=300)
        plt.close()


#end of Evaluator class

def read(file):
    df=pd.read_csv(file)
    return df

def split_data(X:pd.DataFrame, y:pd.DataFrame, groupName:str):
    """Perform a grouped shuffle split on the whole data(always the same with random_state=42).\n
    Grouping is done on the column specified. The split is 80%/20%.
    :param X: x values from the dataset, including the grouping column.
    :param y: y values (labels) from the dataset.
    :param groupName: Name of the column from the x-values to group by.
    :return Split: xTrain, yTrain, xTest, yTest dataframes in this order."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    trainIdx, testIdx = next(gss.split(X, y, groups=X[groupName]))
    xTrain = X.iloc[trainIdx]
    yTrain = y.iloc[trainIdx]
    xTest = X.iloc[testIdx]
    yTest = y.iloc[testIdx]
    return xTrain, yTrain, xTest, yTest

#FOR NOW: just read the metadata file, group by patient id and test some classifiers

metadataFile="../data/metadata.csv"

df=read(metadataFile)
#Just for the metadata, dropping not usable data, turning boolean values to 0-1, and making the str-values to numeric values
df=df.drop(axis=1,labels=['diameter_1','diameter_2','lesion_id','smoke','drink','background_father','background_mother','pesticide','gender','skin_cancer_history','cancer_history','has_piped_water','has_sewage_system','fitspatrick','img_id'])
df['itch'] = df['itch'].astype(bool).astype(int)
df['grew'] = df['grew'].astype(bool).astype(int)
df['hurt'] = df['hurt'].astype(bool).astype(int)
df['changed'] = df['changed'].astype(bool).astype(int)
df['bleed'] = df['bleed'].astype(bool).astype(int)
df['elevation'] = df['elevation'].astype(bool).astype(int)
df['biopsed'] = df['biopsed'].astype(bool).astype(int)
df = pd.get_dummies(df, columns=['region'],dtype=int)

#EXAMPLE USAGE OF THE EVALUATOR CLASS

#prepare necessary input:
#X_train, y_train & patientGroup of training/working data
y = (df['diagnostic'] == "MEL")#obtain true label column and set it to 0 for non-melanoma and 1 for melanoma
X = df.drop(['diagnostic'], axis=1)#obtain X-data by dropping true label -> BUT it still contains the patient_id because it needs to be part of the split
X_train, y_train, X_test, y_test = split_data(X, y, groupName="patient_id")
patientGroup=X_train["patient_id"]#obtain grouping column for training/working data (grouping by patient_id) (NOT over the whole dataset but only over the training data)
X_train = X_train.drop(["patient_id"], axis=1)#get rid of patient_id in training/working X-data
X_test = X_test.drop(["patient_id"], axis=1)#get rid of patient_id in test X-data

#test different classifiers on the training/working data:
clf1 = RandomForestClassifier()
clf2 = DecisionTreeClassifier()
clf3 = KNeighborsClassifier()

voting_clf = VotingClassifier(estimators=[
    ('rf', clf1), 
    ('dt', clf2), 
    ('knn', clf3)
    ], voting='soft') # or voting='hard'

eval = Evaluator()
eval.evalClassifier(clf1, "RandomForest", X_train, y_train, patientGroup, threshold=0.5)
eval.evalClassifier(clf2, "DecisionTree", X_train, y_train, patientGroup, threshold=0.5, saveCurveROC=True)
eval.evalClassifier(clf3, "KNN", X_train, y_train, patientGroup, threshold=0.5)
eval.evalClassifier(voting_clf, "Voting", X_train, y_train, patientGroup, threshold=0.5)
eval.printPerformances()
eval.makeBoxplot("recall")

#NOTE: We could try other stuff here like different parameters for K in KNN
#      or different max_depth for Tree or Forest and compare the performances.
#      Also we could try shrinking the training data and seeing if we can still
#      get acceptable performance with notably less training data
#      (could plot trainSize vs. performance on training data & test data)
#      For the report we could use the output to conduct a statistical test
#      whether one method is better than the other at some confidence level.