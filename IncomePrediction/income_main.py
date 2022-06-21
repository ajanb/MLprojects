import os

import plotter as vs 

import numpy as np 
import pandas as pd 

from sklearn.model_selection import train_test_split
from time import time 
import matplotlib.pyplot as plt 
import seaborn as sns 


os.chdir("/Users/bekzatajan/Projects/MLprojects/IncomePrediction")

#The last column "income" will be our target label in order to find if the person makes more than $50,000 annually
data = pd.read_csv("../data/census.csv")
print(data.head())

data.info()

#Number of records
num_records = len(data)

#Number of records with an income >= $50k
num_plus50k = len(data[data['income'] == '>50K'])

#Number of records with an income < $50k 
num_minus50k = len(data[data['income'] == '<=50K'])

#Percentage of >$50k income 
plus_percent = 100 * num_plus50k / num_records 

print(
    "\n\nNumber of records: {}\n>$50K: {}\n<50K: {}\nPercentage of >50K: {}"\
    .format(num_records, num_plus50k, num_minus50k, plus_percent)
    )

# Let's look at sex and education levels:
sns.set(style='whitegrid', color_codes=True)

fig=sns.catplot(
    x='sex', 
    col='education_level', 
    data=data, 
    hue='income', 
    kind='count', 
    col_wrap=4,
    )

fig.savefig('Plots/SexEducation.png')

#Assign feature and target values:
raw_income = data['income']
raw_feature = data.drop('income', axis=1)

#Show skewed continuous features of original data
vs.distribution('DistributionRaw',data)

#Log transform the skewed featurs:
skewed = [
    'capital-gain', 
    'capital-loss',
    ]

raw_feature[skewed] = data[skewed].apply(lambda x: np.log(x+1))

#Show log distributions
vs.distribution('DistributionLog', raw_feature, transformed = True)


#Normalize numerical features

from sklearn.preprocessing import MinMaxScaler 

scaler = MinMaxScaler() 
numerical = [
    'age', 'education-num', 'capital-gain',
    'capital-loss', 'hours-per-week',
    ]

raw_feature[numerical] = scaler.fit_transform(data[numerical]) 

print(raw_feature.head())


#Data preprocessing

features=pd.get_dummies(raw_feature)

income = raw_income.apply(lambda x: 1 if x == '>50K' else 0) 

encoded = list(features.columns)

print(
    '{} total features after one-hot encoding'\
    .format(len(encoded))
    )


# Shuffle and split data 

X_train, X_test, y_train, y_test = train_test_split(
                                    features, 
                                    income, 
                                    test_size=0.2, 
                                    random_state=0,
                                    )

print('Training set shape: {}'.format(X_train.shape))
print('Test set shape: {}'.format(X_test.shape))


# Evaluate the model performance 

accuracy = num_plus50k / num_records 
precision = num_plus50k / (num_plus50k + num_minus50k)
recall = num_plus50k / (num_plus50k + 0)
f1score = (1 + (0.5 * 0.5)) * (precision * recall / (0.5 * 0.5 * precision +recall))

print(
    'Naive predictor:\n\tAccuracy score: {:.4f}\n\tF1-score: {:.4f}'\
    .format(accuracy, f1score)
    )




