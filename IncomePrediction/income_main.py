import numpy as np 
import pandas as pd 
from time import time 
import os
import seaborn as sns 
import matplotlib.pyplot as plt 
os.chdir("/Users/bekzatajan/Projects/MLprojects/IncomePrediction")
import plotter as vs 


#The last column "income" will be our target label in order to find if the person makes more than $50,000 annually
data = pd.read_csv("../data/census.csv")
print(data.head())

data.info()

#Number of records
num_records = len(data)

#Number of records with an income >= $50k
num_plus50k = len(data[data['income']=='>50K'])

#Number of records with an income < $50k 
num_minus50k = len(data[data['income']=='<=50K'])

#Percentage of >$50k income 
plus_percent = 100 * num_plus50k / num_records 

print("\n\nNumber of records: {}\n>$50K: {}\n<50K: {}\nPercentage of >50K: {}".format(num_records, num_plus50k, num_minus50k, plus_percent))

# Let's look at sex and education levels:
sns.set(style='whitegrid', color_codes=True)
fig=sns.catplot(x='sex', col='education_level', data=data, hue='income', kind='count', col_wrap=4)
fig.savefig('Plots/SexEducation.png')

#Assign feature and target values:
raw_income = data['income']
raw_feature = data.drop('income', axis=1)

#Show skewed continuous features of original data
vs.distribution('DistributionRaw',data)

#Log transform the skewed featurs:
skewed = ['capital-gain', 'capital-loss']
raw_feature[skewed] = data[skewed].apply(lambda x: np.log(x+1))

#Show log distributions
vs.distribution('DistributionLog', raw_feature, transformed=True)


#Normalize numerical features

from sklearn.preprocessing import MinMaxScaler 

scaler = MinMaxScaler() 
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
raw_feature[numerical] = scaler.fit_transform(data[numerical]) 

print(raw_feature.head())



