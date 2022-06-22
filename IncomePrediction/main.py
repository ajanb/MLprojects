import os

import plotter as vs 

import numpy as np 
import pandas as pd 

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.metrics import confusion_matrix

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
fscore = (1 + (0.5 * 0.5)) * (precision * recall / (0.5 * 0.5 * precision +recall))

print(
    'Naive predictor:\n\tAccuracy score: {:.4f}\n\tF1-score: {:.4f}'\
    .format(accuracy, f1score)
    )

# Decision Trees 

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):

    """
    inputs:
        - learner: the learning algorithm to be trained and predicted on 
        - sample_size: the size of samples to be drawn from training set 
        - X_train: features of training set 
        - y_train: target of training set
        - X_test: features of testing set 
        - y_test: target of tetsting set 
    """
    
    results = {}
    
    # Fit the learner to the training data using slicing with 'sample_size'
    start = time()
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() 
    
    # Calculate the training time 
    results['train_time'] = end - start 
    
    # Get the predictions on the test set 
    # Get predictions on the first 300 training samples 
    start = time() 
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time()

    # Calculate the total prediction time 
    results['pred_time'] = end - start

    # Calculate accuracy on the first 300 training samples 
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)

    # Calculate accuracy on the test set 
    results['acc_test'] = accuracy_score(y_test, predictions_test)

    # Calculate F-score on the first 300 training samples
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta=0.5)

    # Calculate F-score on the testing set 
    results['f_test'] = fbeta_score(y_test, predictions_test, beta=0.5)

    # Success
    print('{} trained on {} samples.'.format(learner.__class__.__name__, sample_size))

    return results 
    


# Model Evaluation

# Initiate 3 models
clf_A = DecisionTreeClassifier(random_state=111)
clf_B = SVC(random_state=111)
clf_C = AdaBoostClassifier(random_state=111)

samples_1 = int(round(len(X_train) / 100))
samples_10 = int(round(len(X_train) /10))
samples_100 = len(X_train)

# Collect results on the learners
results = {}

for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = train_predict(
            clf, samples, 
            X_train, 
            y_train, 
            X_test, 
            y_test,
        )

vs.evaluate('evaluateClassifiers', results, accuracy, fscore)


# Print out the values
for i in results.items():
    print(i[0])
    display(pd.DataFrame(i[1]).rename(columns={
                                          0: '1%',
                                          1: '10%', 
                                          2: '100%',
                                      }))


# Visualize the confusion matrix for each classifier 
for i, model in enumerate([clf_A, clf_B, clf_C]):
    cm = confusion_matrix(y_test, model.predict(X_test))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(i)
    sns.heatmap(
        cm, 
        annot = True, 
        annot_kws={'size': 30},
        cmap='Blues',
        square = True,
        fmt='.3f',
    )

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion matrix for: \n{}'.format(model.__class__.__name__))
    plt.savefig('Plots/ConfusionMatrixFor_'+model.__class__.__name__+'.png')




