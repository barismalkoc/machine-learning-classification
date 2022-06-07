# -*- coding: utf-8 -*-
"""
Created on Fri May 27 21:10:29 2022

@author: baris
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay




# Verinin Okunması 

dataSet = pd.read_csv('diabetes-dataset.csv')

# Veri İle İlgili Genel Bilgiler

print("*** NaN değerler ***")
print(dataSet.isnull().sum().sort_values(ascending=False))


# Eksik Verilerin Yok Edilmesi ve Birleştirilmesi

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
outCome = dataSet['Outcome']

dataWithoutNaN = dataSet.drop(['DiabetesPedigreeFunction','Glucose','Insulin'],axis=1)

insulin = dataSet.iloc[:,4:5]
insulin = imputer.fit_transform(insulin)
insulinData = pd.DataFrame(data=insulin, index=range(2002),columns=['Insulin'])


diabetesPedigreeFunction = dataSet.iloc[:,6:7]
diabetesPedigreeFunction = imputer.fit_transform(diabetesPedigreeFunction)
diabetesPedigreeFunctionData = pd.DataFrame(data=diabetesPedigreeFunction, index=range(2002),columns=['DiabetesPedigreeFunction'])

glucose = dataSet.iloc[:,1:2]
glucose = imputer.fit_transform(glucose)
glucoseData = pd.DataFrame(data=glucose, index=range(2002),columns=['Glucose'])

completedData = pd.concat([insulinData,diabetesPedigreeFunctionData,glucoseData, dataWithoutNaN], axis=1)

print("*** NaN değerler Son Hali ***")
print(completedData.isnull().sum().sort_values(ascending=False))

X = completedData.iloc[:,:8].values
y = completedData.iloc[:,8:9].values.ravel()

# eğitim ve test kümelerinin bölünmesi 

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=0)

# standartizasyon

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# PCA Uygulanması

from sklearn.decomposition import PCA
pca = PCA(n_components=7)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


# Logistic Regression
from sklearn.linear_model import LogisticRegression

#PCA olmadan LR
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test,y_pred, labels=classifier.classes_)

matrix_lr = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classifier.classes_)
matrix_lr.plot()
plt.title('PCA Olmadan Lojistik Regresyon')
plt.show()



#PCA ile LR
classifierWithPCA = LogisticRegression(random_state=0)
classifierWithPCA.fit(X_train_pca,y_train)
y_predWithPCA = classifierWithPCA.predict(X_test_pca)
cmWithPCA = confusion_matrix(y_test,y_predWithPCA)

matrix_lr_pca = ConfusionMatrixDisplay(confusion_matrix=cmWithPCA,display_labels=classifierWithPCA.classes_)
matrix_lr_pca.plot()
plt.title('PCA İle Lojistik Regresyon')
plt.show()


#Decision Tree 
from sklearn.tree import DecisionTreeClassifier

#PCA olmadan DR
dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train,y_train)
y_predForDeTree = dtc.predict(X_test)
cmForDecisionTree = confusion_matrix(y_test,y_predForDeTree,labels=dtc.classes_)
matrix_dr = ConfusionMatrixDisplay(confusion_matrix=cmForDecisionTree,display_labels=dtc.classes_)
matrix_dr.plot()
plt.title('PCA Olmadan Karar Ağacı')
plt.show()

print('PCA Olmadan DR r2')
print(r2_score(y_test, y_predForDeTree))

#PCA ile DR

dtc_pca = DecisionTreeClassifier(criterion='entropy')
dtc_pca.fit(X_train_pca,y_train)
y_predForDeTreewithPCA = dtc_pca.predict(X_test_pca)
cmForDecisionTreeWithPCA = confusion_matrix(y_test,y_predForDeTreewithPCA,labels=dtc_pca.classes_)

matrix_dr_pca = ConfusionMatrixDisplay(confusion_matrix=cmForDecisionTreeWithPCA,display_labels=dtc_pca.classes_)
matrix_dr_pca.plot()
plt.title('PCA İle Karar Ağacı')
plt.show()

print('PCA İle DR r2')
print(r2_score(y_test, y_predForDeTreewithPCA))

    
# RandomForestClass
from sklearn.ensemble import RandomForestClassifier

#PCA olmadan RF
rfc = RandomForestClassifier(n_estimators=11, criterion = 'entropy')
rfc.fit(X_train,y_train)
y_predForRandomForest = rfc.predict(X_test)
cmForRandomForest = confusion_matrix(y_test,y_predForRandomForest)

matrix_rf = ConfusionMatrixDisplay(confusion_matrix=cmForRandomForest,display_labels=rfc.classes_)
matrix_rf.plot()
plt.title('PCA Olmadan Random Forest CM')
plt.show()

print('PCA Olmadan RF r2')
print(r2_score(y_test, y_predForRandomForest))


#PCA ile RF
rfc_pca = RandomForestClassifier(n_estimators=11, criterion = 'entropy')
rfc_pca.fit(X_train_pca,y_train)
y_predForRandomForestWithPCA = rfc_pca.predict(X_test_pca)
cmForRandomForestWithPCA = confusion_matrix(y_test,y_predForRandomForestWithPCA)

matrix_rf_pca = ConfusionMatrixDisplay(confusion_matrix=cmForRandomForestWithPCA,display_labels=rfc_pca.classes_)
matrix_rf_pca.plot()
plt.title('PCA İle Random Forest CM')
plt.show()

print('PCA İle RF r2')
print(r2_score(y_test, y_predForRandomForestWithPCA))


















