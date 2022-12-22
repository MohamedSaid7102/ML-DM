from pandas import read_csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

dataset = read_csv('final.csv')
## Male = 0   Female = 1
dataset['gender'].replace("Male",0,inplace=True)
dataset['gender'].replace("Female",1,inplace=True)
dataset['gender'].replace("Other",0,inplace=True)
## Yes = 1     No =0
dataset['ever_married'].replace("Yes",1,inplace=True)
dataset['ever_married'].replace("No",0,inplace=True)
##  private = 0   self_emp = 1   Govt_job = 2   children = 3  Never_worked =4
dataset['work_type'].replace("Private",0,inplace=True)
dataset['work_type'].replace("Self-employed",1,inplace=True)
dataset['work_type'].replace("Govt_job",2,inplace=True)
dataset['work_type'].replace("children",3,inplace=True)
dataset['work_type'].replace("Never_worked",4,inplace=True)
## Urban = 1   Rural=0
dataset['Residence_type'].replace("Urban",1,inplace=True)
dataset['Residence_type'].replace("Rural",0,inplace=True)
## formerly smoked =1  never smoked =0  smokes =2  Unknown=3
dataset['smoking_status'].replace("formerly smoked",1,inplace=True)
dataset['smoking_status'].replace("never smoked",0,inplace=True)
dataset['smoking_status'].replace("smokes",2,inplace=True)
dataset['smoking_status'].replace("Unknown",3,inplace=True)

## Split
X = dataset.iloc[:, 2 : 12]
Y = dataset.iloc[:, 12]
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=42)

## KNN
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print("KNN Accuracy:",accuracy_score(y_test,pred))

## SVM
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("SVM Accuracy:", metrics.accuracy_score(y_test, y_pred))






