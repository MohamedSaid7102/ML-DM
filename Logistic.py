from builtins import type

import pandas as pd
from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tkinter as tk

from tkinter import ttk

dataset = read_csv('final.csv')


dataset['gender'].replace("Male", 0, inplace=True)
dataset['gender'].replace("Female", 1, inplace=True)
dataset['gender'].replace("Other", 0, inplace=True)
## Yes = 1     No =0
dataset['ever_married'].replace("Yes", 1, inplace=True)
dataset['ever_married'].replace("No", 0, inplace=True)
##  private = 0   self_emp = 1   Govt_job = 2   children = 3  Never_worked =4
dataset['work_type'].replace("Private", 0, inplace=True)
dataset['work_type'].replace("Self-employed", 1, inplace=True)
dataset['work_type'].replace("Govt_job", 2, inplace=True)
dataset['work_type'].replace("children", 3, inplace=True)
dataset['work_type'].replace("Never_worked", 4, inplace=True)
## Urban = 1   Rural=0
dataset['Residence_type'].replace("Urban", 1, inplace=True)
dataset['Residence_type'].replace("Rural", 0, inplace=True)
## formerly smoked =1  never smoked =0  smokes =2  Unknown=3
dataset['smoking_status'].replace("formerly smoked", 1, inplace=True)
dataset['smoking_status'].replace("never smoked", 0, inplace=True)
dataset['smoking_status'].replace("smokes", 2, inplace=True)
dataset['smoking_status'].replace("Unknown", 3, inplace=True)


#
# labelencoder_X = LabelEncoder()
##split data
X = dataset.iloc[:, 2 : 12]
Y = dataset.iloc[:, 12]

# X = X.apply(LabelEncoder().fit_transform)



X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=42)

## train data
LogReg = LogisticRegression( solver='sag',max_iter=1000 ,C =1.0)
LogReg.fit(X_train , y_train)

## predict
y_pred = LogReg.predict(X_test)
y_pred_prob = LogReg.predict_proba(X_test)

## Acc score
AccScore = accuracy_score(y_test, y_pred )
print("Accuracy Score is :", AccScore)
print(X.iloc[0, 0 : 10])

##################################
########  Our GUI   ##############
##################################


win = tk.Tk()

win.title('ML')
# Start of Global vars
inputWidth = 26
buttonWidth = 26
#
GenderInitial = 'Male'
AgeInitial = 67.0
HypertensionInitial = 0
HeartDiseaseInitial = 1
EverMarriedInitial = 'Yes'
WorkTypeInitial = 'Private'
ResidenceTypeInitial = 'Urban'
AvgGlucoseLevelInitial = 228.69
BmiInitial = 36.6
SmokingStatusInitial = 'formerly smoked'
# End of Global var

#Gender
Gender=ttk.Label(win,text="Gender")
Gender.grid(row=0,column=0,sticky=tk.W)
Gender_var=tk.StringVar()
Gender_entrybox=ttk.Entry(win,width=inputWidth,textvariable=Gender_var)
Gender_entrybox.grid(row=0,column=1)
Gender_entrybox.insert(0, GenderInitial)


#Age
Age=ttk.Label(win,text="Age")
Age.grid(row=1,column=0,sticky=tk.W)
Age_var=tk.StringVar()
Age_entrybox=ttk.Entry(win,width=inputWidth,textvariable=Age_var)
Age_entrybox.grid(row=1,column=1)
Age_entrybox.insert(0, AgeInitial)

#Hypertension 3
Hypertension=ttk.Label(win,text="Hypertension")
Hypertension.grid(row=2,column=0,sticky=tk.W)
Hypertension_var=tk.StringVar()
Hypertension_entrybox=ttk.Entry(win,width=inputWidth,textvariable=Hypertension_var)
Hypertension_entrybox.grid(row=2,column=1)
Hypertension_entrybox.insert(0, HypertensionInitial)

# HeartDisease
HeartDisease=ttk.Label(win,text="HeartDisease")
HeartDisease.grid(row=3,column=0,sticky=tk.W)
HeartDisease_var=tk.StringVar()
HeartDisease_entrybox=ttk.Entry(win,width=inputWidth,textvariable=HeartDisease_var)
HeartDisease_entrybox.grid(row=3,column=1)
HeartDisease_entrybox.insert(0, HeartDiseaseInitial)

# EverMarried
EverMarried=ttk.Label(win,text="EverMarried")
EverMarried.grid(row=4,column=0,sticky=tk.W)
EverMarried_var=tk.StringVar()
EverMarried_entrybox=ttk.Entry(win,width=inputWidth,textvariable=EverMarried_var)
EverMarried_entrybox.grid(row=4,column=1)
EverMarried_entrybox.insert(0, EverMarriedInitial)

# WorkType
WorkType=ttk.Label(win,text="WorkType")
WorkType.grid(row=5,column=0,sticky=tk.W)
WorkType_var=tk.StringVar()
WorkType_entrybox=ttk.Entry(win,width=inputWidth,textvariable=WorkType_var)
WorkType_entrybox.grid(row=5,column=1)
WorkType_entrybox.insert(0, WorkTypeInitial)

# ResidenceType
ResidenceType=ttk.Label(win,text="ResidenceType")
ResidenceType.grid(row=6,column=0,sticky=tk.W)
ResidenceType_var=tk.StringVar()
ResidenceType_entrybox=ttk.Entry(win,width=inputWidth,textvariable=ResidenceType_var)
ResidenceType_entrybox.grid(row=6,column=1)
ResidenceType_entrybox.insert(0, ResidenceTypeInitial)

# AvgGlucoseLevel
AvgGlucoseLevel=ttk.Label(win,text="AvgGlucoseLevel")
AvgGlucoseLevel.grid(row=7,column=0,sticky=tk.W)
AvgGlucoseLevel_var=tk.StringVar()
AvgGlucoseLevel_entrybox=ttk.Entry(win,width=inputWidth,textvariable=AvgGlucoseLevel_var)
AvgGlucoseLevel_entrybox.grid(row=7,column=1)
AvgGlucoseLevel_entrybox.insert(0, AvgGlucoseLevelInitial)

# Bmi
Bmi=ttk.Label(win,text="Bmi")
Bmi.grid(row=8,column=0,sticky=tk.W)
Bmi_var=tk.StringVar()
Bmi_entrybox=ttk.Entry(win,width=inputWidth,textvariable=Bmi_var)
Bmi_entrybox.grid(row=8,column=1)
Bmi_entrybox.insert(0, BmiInitial)

# SmokingStatus
SmokingStatus=ttk.Label(win,text="SmokingStatus")
SmokingStatus.grid(row=9,column=0,sticky=tk.W)
SmokingStatus_var=tk.StringVar()
SmokingStatus_entrybox=ttk.Entry(win,width=inputWidth,textvariable=SmokingStatus_var)
SmokingStatus_entrybox.grid(row=9,column=1)
SmokingStatus_entrybox.insert(0, SmokingStatusInitial)


DF = pd.DataFrame()

def action():
    DF = pd.DataFrame(columns=['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type'
        ,'avg_glucose_level','Bmi','smoking_status'])

    Gender=Gender_var.get()
    DF.loc[0,'gender']=Gender

    Age=Age_var.get()
    DF.loc[0,'age']=Age

    Hypertension=Hypertension_var.get()
    DF.loc[0,'hypertension']=Hypertension

    HeartDisease=HeartDisease_var.get()
    DF.loc[0,'heart_disease']=HeartDisease

    EverMarried=EverMarried_var.get()
    DF.loc[0,'ever_married']=EverMarried

    WorkType=WorkType_var.get()
    DF.loc[0,'work_type']=WorkType

    ResidenceType=ResidenceType_var.get()
    DF.loc[0,'Residence_type']=ResidenceType

    AvgGlucoseLevel=AvgGlucoseLevel_var.get()
    DF.loc[0,'avg_glucose_level']=AvgGlucoseLevel

    Bmi=Bmi_var.get()
    DF.loc[0,'Bmi']=Bmi

    SmokingStatus=SmokingStatus_var.get()
    DF.loc[0,'smoking_status']=SmokingStatus

    # Turn data into numbers
    DF['gender'].replace("Male", 0.00, inplace=True)
    DF['gender'].replace("Female", 1.00, inplace=True)
    DF['gender'].replace("Other", 0.00, inplace=True)
    ## Yes = 1     No =0
    DF['ever_married'].replace("Yes", 1.00, inplace=True)
    DF['ever_married'].replace("No", 0.00, inplace=True)
    ##  private = 0   self_emp = 1   Govt_job = 2   children = 3  Never_worked =4
    DF['work_type'].replace("Private", 0.00, inplace=True)
    DF['work_type'].replace("Self-employed", 1.00, inplace=True)
    DF['work_type'].replace("Govt_job", 2.00, inplace=True)
    DF['work_type'].replace("children", 3.00, inplace=True)
    DF['work_type'].replace("Never_worked", 4.00, inplace=True)
    ## Urban = 1   Rural=0
    DF['Residence_type'].replace("Urban", 1.00, inplace=True)
    DF['Residence_type'].replace("Rural", 0.00, inplace=True)
    ## formerly smoked =1  never smoked =0  smokes =2  Unknown=3
    DF['smoking_status'].replace("formerly smoked", 1.00, inplace=True)
    DF['smoking_status'].replace("never smoked", 0.00, inplace=True)
    DF['smoking_status'].replace("smokes", 2.00, inplace=True)
    DF['smoking_status'].replace("Unknown", 3.00, inplace=True)

    # Check the output
    output = LogReg.predict(DF)

    print(DF.iloc[0, 0 : 10])

    result = ''

    if output == 1:
        result = 'Strock'
    elif output == 0:
        result = 'Non-Strock'

    print(result)

    # Empty last input
    Predict_entrybox.delete(0,500)
    Predict_entrybox.insert(0, result)

Predict_entrybox=ttk.Entry(win,width=buttonWidth)
Predict_entrybox.grid(row=10,column=1)
Predict_entrybox.insert(1,str(''))
Predict_button=ttk.Button(win,text="Predict",command=action)
Predict_button.grid(row=10,column=0)

win.mainloop()