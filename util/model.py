from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
def read(file):
    df=pd.read_csv(file)
    return df


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def model(X_train, y_train,X_test, y_test):
    # Initialize and train the model
    model = RandomForestClassifier(class_weight='balanced')
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate performance
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

df=read('data\metadata.csv')
#Just for the metadata, droping not usable data, turning boolean values to 0-1, and making the str-values to numeric values
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