import matplotlib
from pandas import read_csv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
dataset = read_csv('final.csv')
print(dataset.head())

plt.scatter(dataset.age, dataset.Bmi)
plt.xlabel('age')
plt.ylabel('Bmi')
plt.show()