# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 11:43:14 2023

@author: ramav
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

test_data = pd.read_csv("D:\\Assignments\\Support Vector Machine\\SalaryData_Test(1).csv")
test_data.head()
test_data.shape
test_data.dtypes
test_data.isnull().sum()
test_data.info()


train_data = pd.read_csv("D:\\Assignments\\Support Vector Machine\\SalaryData_Train(1).csv")
train_data.head()
train_data.shape
train_data.dtypes
train_data.isnull().sum()
train_data.info()

df = pd.concat([test_data,train_data],axis=0)
df.head()
df.shape


df.workclass.value_counts()
df.workclass.unique()

df.native.value_counts()
df.workclass.unique()

df.occupation.value_counts()
df.occupation.unique()

df.sex.value_counts()
df.sex.unique()

# EDA
#histogram
sns.histplot(df["workclass"])
df["workclass"].hist()

sns.histplot(df["education"])
df["education"].hist()

sns.histplot(df["hoursperweek"])
df["hoursperweek"].hist()

sns.histplot(df["native"])
df["native"].hist()

sns.histplot(df["capitalloss"])
df["capitalloss"].hist()

sns.histplot(df["capitalgain"])
df["capitalgain"].hist()

t1 = pd.crosstab(index=df["education"],columns=df["workclass"])
t1.plot(kind='bar')

t2 = pd.crosstab(index=df["education"],columns=df["Salary"])
t2.plot(kind='bar')

t3 = pd.crosstab(index=df["sex"],columns=df["race"])
t3.plot(kind='bar')

t4 = pd.crosstab(index=df["maritalstatus"],columns=df["sex"])
t4.plot(kind='bar')

# box plot
sns.boxplot(df["age"],color="blue")    
sns.boxplot(df["workclass"],color="blue")    
sns.boxplot(df["hoursperweek"],color="black")    
sns.boxplot(df["capitalloss"],color="violet")    
sns.boxplot(df["capitalgain"],color="red")    

# removing outliers

df.boxplot("age",vert=False)
Q1=np.percentile(df["age"],25)
Q3=np.percentile(df["age"],75)
IQR=Q3-Q1
LW=Q1-(2.0*IQR)
UW=Q3+(2.0*IQR)
df["age"]<LW
df[df["age"]<LW]
df[df["age"]<LW].shape
df["age"]>UW
df[df["age"]>UW]
df[df["age"]>UW].shape
df["age"]=np.where(df["age"]>UW,UW,np.where(df["age"]<LW,LW,df["age"]))

df.boxplot("education",vert=False)
Q1=np.percentile(df["education"],25)
Q3=np.percentile(df["education"],75)
IQR=Q3-Q1
LW=Q1-(2.0*IQR)
UW=Q3+(2.0*IQR)
df["education"]<LW
df[df["education"]<LW]
df[df["education"]<LW].shape
df["education"]>UW
df[df["education"]>UW]
df[df["education"]>UW].shape
df["education"]=np.where(df["education"]>UW,UW,np.where(df["education"]<LW,LW,df["education"]))

df.boxplot("hoursperweek",vert=False)
Q1=np.percentile(df["hoursperweek"],25)
Q3=np.percentile(df["hoursperweek"],75)
IQR=Q3-Q1
LW=Q1-(2.0*IQR)
UW=Q3+(2.0*IQR)
df["hoursperweek"]<LW
df[df["hoursperweek"]<LW]
df[df["hoursperweek"]<LW].shape
df["hoursperweek"]>UW
df[df["hoursperweek"]>UW]
df[df["hoursperweek"]>UW].shape
df["hoursperweek"]=np.where(df["hoursperweek"]>UW,UW,np.where(df["hoursperweek"]<LW,LW,df["hoursperweek"]))

df.boxplot("capitalgain",vert=False)
Q1=np.percentile(df["capitalgain"],25)
Q3=np.percentile(df["capitalgain"],75)
IQR=Q3-Q1
LW=Q1-(2.0*IQR)
UW=Q3+(2.0*IQR)
df["capitalgain"]<LW
df[df["capitalgain"]<LW]
df[df["capitalgain"]<LW].shape
df["capitalgain"]>UW
df[df["capitalgain"]>UW]
df[df["capitalgain"]>UW].shape
df["capitalgain"]=np.where(df["capitalgain"]>UW,UW,np.where(df["capitalgain"]<LW,LW,df["capitalgain"]))


# correlation
corr = df.corr()  
plt.figure(figsize=(12,10))
sns.heatmap(corr,annot=True,cmap="magma")  # heat map
plt.show()

# with the following function i will select the highly correlated features
def correlation(dataset,threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]) > threshold:
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
        return(col_corr)


corr_features = correlation(df,0.85)
len(set(corr_features))

corr_features # there is no hhighly correlated features

'''
# Check Correlation amoung parameters
corr = df.corr()
fig, ax = plt.subplots(figsize=(8,8))
# Generate a heatmap
sns.heatmap(corr, cmap = 'magma', annot = True, fmt = ".2f")
plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()

'''
from sklearn.preprocessing import LabelEncoder
df = df.apply(LabelEncoder().fit_transform)
df.head()

drop_elements = ['education', 'native', 'Salary']
x = df.drop(drop_elements, axis=1)
y = df['Salary']


#Data partition 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

#support Vector machine

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)

# make predictions
y_pred = svc.predict(x_test)

# summarize the fit of the model
from sklearn import metrics
from sklearn.metrics import accuracy_score,accuracy_score,confusion_matrix
print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


svc = SVC(kernel='rbf',gamma=2, C=1)
svc.fit(x_train, y_train)

# make predictions
prediction = svc.predict(x_test)

# summarize the fit of the model

print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

print("Accuracy:",metrics.accuracy_score(y_test, prediction))
print("Precision:",metrics.precision_score(y_test, prediction))
print("Recall:",metrics.recall_score(y_test, prediction))


svc = SVC(kernel='poly',degree=3,gamma="scale")
svc.fit(x_train, y_train)
# make predictions
prediction = svc.predict(x_test)
# summarize the fit of the model
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

print("Accuracy:",metrics.accuracy_score(y_test, prediction))
print("Precision:",metrics.precision_score(y_test, prediction))
print("Recall:",metrics.recall_score(y_test, prediction))

# Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train,y_train)

#prediction
y_pred_test = logreg.predict(x_test)

print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

print("Accuracy:",metrics.accuracy_score(y_test, prediction))
print("Precision:",metrics.precision_score(y_test, prediction))
print("Recall:",metrics.recall_score(y_test, prediction))

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(x_train,y_train)

y_pred_test = classifier.predict(x_test)

print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

print("Accuracy:",metrics.accuracy_score(y_test, prediction))
print("Precision:",metrics.precision_score(y_test, prediction))
print("Recall:",metrics.recall_score(y_test, prediction))







