from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier #Promising
from sklearn.neighbors import KNeighborsClassifier  #This one is really unaccurate (10%) which is basicly worst than guessing (but to be more exact
                                                    #this method works well on the larger classes, and the Melanoma is not :( )

from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
def read(file):
    df=pd.read_csv(file)
    return df


def split_data(X, y, test_size=0.2, random_state=42):#I think it's super important that random_state=42 never changes cuz otherwise we overfit by looking at test data
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def model(X_train, y_train,X_test, y_test):
    # Initialize and train the model
    clf1 = RandomForestClassifier()
    clf2 = DecisionTreeClassifier()
    clf3 = KNeighborsClassifier()
    clf4 = LogisticRegression(class_weight="balance")

    voting_clf = VotingClassifier(estimators=[
    ('rf', clf1), 
    ('dt', clf2), 
    ('knn', clf3),
    ('lr', clf4)
    ], voting='hard') # or voting='soft'

    voting_clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = voting_clf.predict(X_test)
    # Evaluate performance
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

file='data\metadata.csv'

df=read(file)
#Just for the metadata, dropping not usable data, turning boolean values to 0-1, and making the str-values to numeric values
df=df.drop(axis=1,labels=['diameter_1','diameter_2','patient_id','lesion_id','smoke','drink','background_father','background_mother','pesticide','gender','skin_cancer_history','cancer_history','has_piped_water','has_sewage_system','fitspatrick','img_id'])
df['itch'] = df['itch'].astype(bool).astype(int)
df['grew'] = df['grew'].astype(bool).astype(int)
df['hurt'] = df['hurt'].astype(bool).astype(int)
df['changed'] = df['changed'].astype(bool).astype(int)
df['bleed'] = df['bleed'].astype(bool).astype(int)
df['elevation'] = df['elevation'].astype(bool).astype(int)
df['biopsed'] = df['biopsed'].astype(bool).astype(int)
df = pd.get_dummies(df, columns=['region'],dtype=int)
y = df['diagnostic']
X = df.drop(['diagnostic'], axis=1)
X_train, X_test, y_train, y_test = split_data(X, y)
model(X_train, y_train, X_test, y_test)

#For the normal data
#patient_group=df['pat_les_ID']
#df=df.drop(axis=1,labels=['name', 'patient_ID', 'lesion_ID', 'pat_les_ID'])
#y = df['true_label']
#X = df.drop(['true_label'], axis=1)
#X_train, X_test, y_train, y_test = split_data(X, y)
#GroupShuffleSplit
#gss=GroupShuffleSplit(n_splits=1,test_size=0.2, random_state=42)
#train_ind,test_ind = next(gss.split(X,y,groups=patient_group))
#X_train_gss, X_test_gss = X[train_idx], X[test_idx]
#y_train_gss, y_test_gss = y[train_idx], y[test_idx]
#train_ind_2,test_ind_2 = next(gss.split(X_train_gss,y_train_gss,groups=patient_group))
#X_train_gss_2, X_test_gss_2 = X[train_idx_2], X[test_idx_2]
#y_train_gss_2, y_test_gss_2 = y[train_idx_2], y[test_idx_2]
#model(X_train, y_train, X_test, y_test)
#model(X_train_gss_2, y_train_gss_2, X_test_gss_2, y_test_gss_2)

#It seems like currently the RandomForestClassifier is the best(in the case of hard voting), because the model only agree on something of the RandomForestClassifier
#says yes. On the other hand soft voting could increase the overall accuracy, Maybe on the normal data it will change.