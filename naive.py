from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

dataset=pd.read_csv('final.csv')
labelencoder_X = LabelEncoder()
X = dataset.iloc[:, 2 : 12]
Y = dataset.iloc[:, 12]
X = X.apply(LabelEncoder().fit_transform)

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=42)
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(classification_report(y_test , y_pred))

print("ACC Score : " , accuracy_score(y_test,y_pred))

#print("Accuracy Score is :"+ ac)


#df=pd.DataFrame({'Actual': y_test , 'predicted': y_pred})
#print(df)