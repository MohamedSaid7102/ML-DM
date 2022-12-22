from pandas import read_csv
import pandas as pd

dataset = read_csv('data.csv')
bmi = dataset.loc[:,'bmi']
cleaned_bmi =[]
default_bmi = list(bmi.mode())[0]

for i in bmi:
    if str(i) == 'NaN' or str(i) == 'nan':
        cleaned_bmi.append(default_bmi)
    else:
        cleaned_bmi.append(i)

output = pd.DataFrame({'id':dataset.loc[:,"id"], 'gender':dataset.loc[:,'gender'] ,'age':dataset.loc[:,"age"],'hypertension':dataset.loc[:,"hypertension"],'heart_disease':dataset.loc[:,'heart_disease'] , 'ever_married':dataset.loc[:,'ever_married'],'work_type':dataset.loc[:,'work_type'],'Residence_type':dataset.loc[:,'Residence_type'],'avg_glucose_level':dataset.loc[:,'avg_glucose_level'],'Bmi':cleaned_bmi ,'smoking_status':dataset.loc[:,'smoking_status'] ,'stroke':dataset.loc[:,'stroke'] })
output.to_csv('final.csv')
















