import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from pandas import read_csv
dataset = read_csv('final.csv')

#dataset = pd.get_dummies(dataset, drop_first=True)
labelencoder_X = LabelEncoder()
##split data
X = dataset.iloc[:, 2 : 12]
Y = dataset.iloc[:, 12]
X = X.apply(LabelEncoder().fit_transform)

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=42)

## train data  build model

clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

##############################3
