import numpy as np 
import pandas as pd 
from time import time 
import os
import seaborn as sns 
import matplotlib.pyplot as plt 

os.chdir("/Users/bekzatajan/Projects/MLprojects/IncomePrediction")

#The last column "income" will be our target label in order to find if the person makes more than $50,000 annually
data = pd.read_csv("../data/census.csv")
data.head()

data.info()

#Number of records
num_records = len(data)

#Number of records with an income >= $50k
num_plus50k = len(data[data['income']=='>50K'])

#Number of records with an income < $50k 
num_minus50k = len(data[data['income']=='<=50K'])

#Percentage of >$50k income 
plus_percent = 100 * num_plus50k / num_records 

print("Number of records: {}\n>$50K: {}\n<50K: {}\nPercentage of >50K: {}".format(num_records, num_plus50k, num_minus50k, plus_percent))

sns.set(style='whitegrid', color_codes=True)
fig=sns.catplot(x='sex', col='education_level', data=data, hue='income', kind='count', col_wrap=4)
fig.savefig('Plots/out.png')



