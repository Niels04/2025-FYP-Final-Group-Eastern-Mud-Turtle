import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.utils import resample

from util.classifier import makeConfusionMatrix, printCrossValidationPerformance#use this for the final submit
#from classifier import makeConfusionMatrix, printCrossValidationPerformance#this is only for running this file on its own

class Formula():
    """Class to implement the classifier based on a set formula.
    Given a dataset, it can predict the label of each datapoint.
    After doing so, it is possible to compute and return precision
    and recall score of the classifier with the relative functions."""
    
    def __init__(self):
        self.preds = None

    def formula(self, datapoint, return_value= False):
        """Given a datapoint, extracts the needed column values to
        compute the formula value, which is then converted into a label.
        By default, only the label is returned, but it is possible to 
        return both label and formula value by setting the parameter
        return_value to True.
        
        :param datapoint: The datapoint to be analysed.
        :param return_value: Boolean value that indicates wether or not the
                            formula value will be returned alongside the label.
        
        :return: The computed label, and optionally the formula value.
        
        """

        # fixed formula
        value = (1.3 * datapoint['A_val']) + (0.1 * datapoint['B_val']) + (0.5 * datapoint['C_val']) + (0.5 * datapoint['D_val'])
        label = 1 if value > 4.7 else 0

        if return_value:
            return label, value
        return label
    
    def predict(self, data: pd.DataFrame):
        """Given a set of datapoints, predicts the label for each.
        Predictions are stored within the class, and also returned
        by the function.
        
        :param data: The dataframe to be analysed.
        
        :return: The series of predictions.
        
        """

        pred = data.copy()

        self.preds = pred.apply(self.formula, axis= 1)

        return self.preds

    def predictions(self):
        """If the model already predicted some labels, this function can be used to
        obtain said predictions.
        
        """
        return self.preds   
     
    def precision(self, true_labels):
        """After having predicted some labels, returns the precision score of
        said predictions based on the given true labels.
        
        :param true_labels: The series of true labels to be used to compute
                            the precision score.
        
        :return: The precision score.
        
        """
        return precision_score(true_labels, self.preds)
    
    def recall(self, true_labels):
        """After having predicted some labels, returns the recall score of
        said predictions based on the given true labels.
        
        :param true_labels: The series of true labels to be used to compute
                            the recall score.
        
        :return: The recall score.
        
        """
        return recall_score(true_labels, self.preds)
    
    def finalCrossValidate(self, x: pd.DataFrame, y: pd.DataFrame, nStraps = 20) -> None:
        """Cross validation via bootstrapping to estimate performance metrics and variances for our method.\n

        :param x: Test x data (features)
        :param y: y True labels, used to calculate performance metrics

        :return None:
        
        """

        # store performances for individual straps
        PREs = np.zeros(nStraps)   #  precision scores
        RECs = np.zeros(nStraps)   #  recall scores

        # store all true labels and predictions
        allYLabels = []
        allYPredictions = []

        for i in range(nStraps):

            x_b, y_b = resample(x, y) # generate a bootstrap by resampling from the test data WITH REPLACEMENT

            self.predict(x_b) # get predicted probabilities from model

            # compute precision and recall for current strap
            PREs[i] = self.precision(y_b)
            RECs[i] = self.recall(y_b)

            # save prediction probabilites and true labels
            allYPredictions.extend(self.predictions()) # save predictions for current strap
            allYLabels.extend(y_b) # save true labels for current strap

        # create confusion matrix
        makeConfusionMatrix("Formula", allYLabels, allYPredictions, dataType= "test", combined= nStraps)

        # finally print the performances
        print(f"Performance for cross validation for method Formula")
        printCrossValidationPerformance("precision", PREs)
        printCrossValidationPerformance("Recall", RECs)
        
        #do one final predict run with all the test data to get a concrete confusion matrix
        yPred = self.predict(x)#get predicted labels from model
        makeConfusionMatrix("Formula", y, yPred, dataType="test")#make confusion matrix showing raw performance on test data
    
    def runFormulaClassifier(self, x: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        """Given required data, perform a single run of the formula classifier\n
        and if true melanoma label is available for the test data, also compute some test statistics.\n

        :param x: Data frame containing features needed to compute the formula value.
        :param y: (optional) Data frame containing true labels.

        :return result: pandas dataframe with [\"img_id\", \"melanoma_prediction\"] + [\"true_melanoma_label\"] (if available)
        
        """

        self.predict(x)

        # create result dataframe to store img_id and prediction (+ true label if available)
        result = pd.DataFrame({
        'img_id': x["img_id"],
        'melanoma_prediction': self.predictions()
        })

        if y is not None:

            # add true melanoma labels to result dataframe:
            result["true_melanoma_label"] = y

            # PERFORM CROSS VALIDATION
            self.finalCrossValidate(x, y)

        return result