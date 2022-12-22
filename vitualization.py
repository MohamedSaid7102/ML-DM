from pandas import read_csv
from matplotlib import pyplot
from seaborn import lineplot, distplot, scatterplot,barplot, boxplot

dataset = read_csv('final.csv')
#lineplot(data=dataset)
#distplot(a = dataset)               #bins
#barplot(data=dataset)
#scatterplot(data=dataset)
boxplot(data=dataset["Bmi"])
pyplot.show()


