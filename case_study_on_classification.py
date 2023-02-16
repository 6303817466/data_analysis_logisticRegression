# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 16:13:55 2023

@author: Mohan
"""
import os

import pandas as pd

import numpy as np

import seaborn as sns

#to patition the data
from sklearn.model_selection import train_test_split

#importing library for Logistic Regression
from sklearn.linear_model import LogisticRegression

#importing performance metric -accuracy score and confusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix


os.chdir('D:\pandas')

data_income=pd.read_csv('income.csv')

data=data_income.copy()

print(data.info())

data.isnull()

print('Data coloumns with null values :\n',data.isnull().sum())

summary_num=data.describe()
print(summary_num)

summary_cate=data.describe(include="O")
print(summary_cate)

print(data['JobType'].value_counts())
print(data['occupation'].value_counts())


print(np.unique(data['JobType']))
print(np.unique(data['occupation']))


data=pd.read_csv('income.csv',na_values=[" ?"])

print(data.isnull().sum())

missing = data[data.isnull().any(axis=1)]

data2=data.dropna(axis=0)

correlation_num=data2.corr()

data2.columns

print(data2.dtypes)


gender=pd.crosstab(index=data2['gender'],
                   columns="count",
                   normalize=True)
print(gender)

#gender vs salary

gender_salstat=pd.crosstab(index=data2['gender'],
                           columns=data2['SalStat'],
                           margins=True,
                           normalize='index')
print(gender_salstat)

sns.countplot(y='JobType',data=data2,
              hue='SalStat')

sns.boxplot(x='SalStat',y='hoursperweek',data=data2,hue='SalStat')

#LOGISTIC REGRESSION

data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

cols=['gender','nativecountry','race','JobType']
new_data=data2.drop(cols,axis=1)

new_data=pd.get_dummies(data2,drop_first=True)

column_list=list(new_data.columns)
print(column_list)


features=list(set(column_list)-set(['SalStat']))
print(features)

y=new_data['SalStat'].values
print(y)

x=new_data[features].values
print(x)

train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3,random_state=False)

logistic=LogisticRegression()

logistic.fit(train_x,train_y)

prediction = logistic.predict(test_x)

accuracy_score = accuracy_score(test_y, prediction)
print(accuracy_score)

print('Misclassified sample : %d'%(test_y!=prediction).sum())

