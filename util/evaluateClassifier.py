import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

class Evaluator:
    def __init__(self):
        self.accuracies = {}
    
    #NOTE: This is for our own iterative process of finding the best classifier/trainSize/hyperparameters like tree depth, etc...
    #      For the final evaluation we could slightly re-write this method to measure prediction accuracy on actual test data
    #      (not validation data like its using right now)
    #      BUT I believe we shouldn't iterate & change our model after running it on the actual test data
    #      (would be Overfitting by Observer as described in the lecture)
    #      so I think it's super important that we keep "random_state=42" in the split_data the whole time

    #      To be added:
    #       - automatic boxplot generation of performances for all inputed classifiers (should maybe use higher nShuffles)
    #       - method that generates ROC curve graphic for a selected classifier
    #       - method that evaluates performance with test data (only use at end) -> could use for hypothesis testing
    #       - 
    def evalClassifier(self, classifier, name:str, xTrain:pd.DataFrame, yTrain:pd.DataFrame, patientGroups:pd.DataFrame, nShuffles = 5, validationSize=0.2):
        """Given the classifier, computes AUC(accuracy) over given
        number of grouped shuffles of the given training data,
        grouped by the given column.\n
        Stores results in Evaluator class (can be printed with
        \"printPerformances()\").

        :param classifer: The untrained classifier to be evaluated
        :param name: Classifier name string
        :param xTrain: The x-columns of the training/working data
        :param yTrain: The y-column of the training/working data
        :param patientGroups: Column of trainging/working data to group by
        :param nShuffles: Number of different shuffled folds to perform, default=5
        :param validationSize: Proportion of training/working data to be used as validation Data, default=0.2
        :return None:"""
        ACCs = np.zeros(nShuffles)#array to store accuracies(AUCss)
        
        #iterate through all splits, train the model and evaluate performance
        gss=GroupShuffleSplit(n_splits=nShuffles, test_size=validationSize, random_state=42)
        for i, (trainIdx, valIdx) in enumerate(gss.split(xTrain, yTrain, patientGroups)):
            #fit classifier on current split
            classifier.fit(xTrain.iloc[trainIdx], yTrain.iloc[trainIdx])
            #test classifier on validation data for current split
            yPred = classifier.predict(xTrain.iloc[valIdx])
            ACCs[i] = accuracy_score(yTrain.iloc[valIdx], yPred)

        self.accuracies[name] = (np.mean(ACCs), np.var(ACCs))#store performance of the classifier in dictionary
    
    def printPerformances(self) -> None:
        for name, acc in self.accuracies.items():
            print(f"Accuracy for classifier \"{name}\"")
            print(f"Mean: {acc[0]}")#print mean accuracy over shouffles for this method
            print(f"Variance: {acc[1]}")#print variance of accuracy over shouffles for this method
            print("\n")
#end of Evaluator class

def read(file):
    df=pd.read_csv(file)
    return df

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

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
y = df['diagnostic']#obtain true label column
X = df.drop(['diagnostic'], axis=1)#obtain X-data by dropping true label -> BUT it still contains the patient_id because it needs to be part of the split
X_train, X_test, y_train, y_test = split_data(X, y)
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
    ], voting='hard') # or voting='soft'

eval = Evaluator()
eval.evalClassifier(clf1, "RandomForest", X_train, y_train, patientGroup)
eval.evalClassifier(clf2, "DecisionTree", X_train, y_train, patientGroup)
eval.evalClassifier(clf3, "KNN", X_train, y_train, patientGroup)
eval.evalClassifier(voting_clf, "Voting", X_train, y_train, patientGroup)
eval.printPerformances()

#NOTE: We could try other stuff here like different parameters for K in KNN
#      or different max_depth for Tree or Forest and compare the performances.
#      Also we could try shrinking the training data and seeing if we can still
#      get acceptable performance with notably less training data
#      (could plot trainSize vs. performance on training data & test data)
#      For the report we could use the output to conduct a statistical test
#      whether one method is better than the other at some confidence level.