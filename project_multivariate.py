#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 19:36:07 2019

@author: lizhiying
"""


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV

from tqdm import tqdm

####read the dataset####
data =  pd.read_csv('/Users/lizhiying/Desktop/Multivariate/bank/raw_data.csv')


#----------------------------
'''EDA'''

sns.set_style("whitegrid")
sns.set_palette(sns.color_palette("Paired"))

ax = plt.axes()
sns.countplot(x='y', data=data, ax=ax)
ax.set_title('Target class distribution')
plt.show()

cols = ['age','balance','day','duration','campaign','pdays','previous','y']
eda = data[cols]




'''Correlation heatmap'''
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(eda.corr(), annot=True, linewidths=.5, fmt= '.3f',ax=ax,cmap = 'Blues')
plt.show()


'''Scatter plot'''
sns.pairplot(eda,diag_kind="kde")

sns.pairplot(eda,hue='y')


'''Paired plot'''
f, axarr = plt.subplots(2, 2, figsize=(15, 15))

sns.boxplot(x='age', y='y', data=data, showmeans=True, ax=axarr[0,0])
sns.boxplot(x='balance', y='y', data=data, showmeans=True, orient = 'h',ax=axarr[0, 1]).set(xscale = "log")
sns.boxplot(x='duration', y='y', data=data, showmeans=True, ax=axarr[1, 0])
sns.boxplot(x='campaign', y='y', data=data, showmeans=True, ax=axarr[1, 1])


axarr[0, 0].set_title('age')
axarr[0, 1].set_title('education')
axarr[1, 0].set_title('duration')
axarr[1, 1].set_title('campaign')
 
#plt.tight_layout()
plt.show()

 
#-------------------------------------------------------------
'''Model'''

df =  pd.read_csv('/Users/lizhiying/Desktop/Multivariate/bank/processed_data.csv',sep = ',')


y = df["Y"]
X = df.ix[:,1:44]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



#-------------------------------------------------------------
'''Logistic''' 

lr = LogisticRegression(solver = 'lbfgs')
parameters = {'C':[0.01,0.1,1,10]}
clf = GridSearchCV(lr, parameters, cv=10)
clf.fit(X_train,y_train)

#clf.cv_results_
plot = sns.barplot(parameters.get('C'),clf.cv_results_['mean_test_score']*100,palette="Blues_d")

plot.set_ylim(88.8,89.3)
plot.set_ylabel("mean test score")
plot.set_xlabel("l2 penalty term")
plot.show()

#-------------------------------------------------------------
'''Random Forest'''

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)

# Feature importance 
feature_importances = pd.DataFrame(rfc.feature_importances_,index = X_train.columns,
                                   columns=['importance']).sort_values('importance', ascending =False)

feature_importances['features'] = feature_importances.index.values



f,ax = plt.subplots(figsize=(15, 15))
sns.set_context("poster",font_scale = 2)
sns.barplot(y="features", x="importance", data=feature_importances.iloc[0:10,:],
            label="Feature Importance", palette="Set2")


#cross validation
rfc = RandomForestClassifier()
parameters = {'n_estimators':[10,100,200],'max_depth':[2,3,4,5]}
clf = GridSearchCV(rfc, parameters, cv=5)
clf.fit(X_train,y_train)

pvt = pd.pivot_table(pd.DataFrame(clf.cv_results_),
    values='mean_test_score', index='param_n_estimators', columns='param_max_depth')

plt.figure(figsize=(10,8))
plt.title('Classification Accuracy with hidden_layer_sizes and activations')
ax = sns.heatmap(pvt,cmap="Greens")
plt.show()


#-----------------------------------------------------
'''SVM'''

svc = SVC(kernel = 'rbf')
parameters = {'C':[0.01,0.1,1]}
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(X_train,y_train)

plot = sns.barplot(parameters.get('C'),clf.cv_results_['mean_test_score']*100,palette="Blues_d")

plot.set_ylim(88.26,88.28)
plot.set_ylabel("mean test score")
plot.set_xlabel("l2 penalty term")







#------------------------------------------------------
''' Neural Network '''

mlp = MLPClassifier(activation='relu', solver='adam')
parameters = {'hidden_layer_sizes':[(5,),(10,),(20,),(50,)],'activation' : ["relu","logistic","tanh"]}
clf = GridSearchCV(mlp, parameters, cv=3)
clf.fit(X_train,y_train)

pvt = pd.pivot_table(pd.DataFrame(clf.cv_results_),
    values='mean_test_score', index='param_hidden_layer_sizes', columns='param_activation')

plt.figure(figsize=(10,8))
plt.title('Classification Accuracy with hidden_layer_sizes and activations')
ax = sns.heatmap(pvt,cmap="Oranges")
plt.show()

#----------------------------------------------------
''' AUC ROC'''

lr = LogisticRegression(solver = 'lbfgs',C = 10).fit(X_train,y_train)
y_pred_lr = lr.predict_proba(X_test)[:,1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test,y_pred_lr)



rfc = RandomForestClassifier(n_estimators=10).fit(X_train,y_train)
y_pred_rf = rfc.predict_proba(X_test)[:,1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test,y_pred_rf)



svc = SVC(kernel = 'rbf',C = 0.01, probability =True).fit(X_train,y_train)
y_pred_sv = svc.predict_proba(X_test)[:,1]
fpr_sv, tpr_sv, _ = roc_curve(y_test, y_pred_sv)
auc_sv = roc_auc_score(y_test,y_pred_sv)



mlp = MLPClassifier(activation='relu', solver='adam',hidden_layer_sizes = (10,)).fit(X_train,y_train)
y_pred_nn = mlp.predict_proba(X_test)[:,1]
fpr_nn, tpr_nn, _ = roc_curve(y_test, y_pred_nn)
auc_nn = roc_auc_score(y_test,y_pred_nn)








plt.subplots(figsize=(8, 8))
plt.xlim(0, 1)
plt.ylim(0, 1)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf,color = 'green',label = 'Random Forest ROC curve (area = {0:0.2f})'
               ''.format(auc_rf),lw=2)
plt.plot(fpr_nn, tpr_nn, label = 'Neural Network ROC curve (area = {0:0.2f})'
               ''.format(auc_nn),lw=2)
plt.plot(fpr_lr, tpr_lr,color = 'red',label = 'Logistic ROC curve (area = {0:0.2f})'''.format(auc_lr),lw=2)

plt.plot(fpr_sv, tpr_sv ,color = 'darkorange',label = 'SVM ROC curve (area = {0:0.2f})'
             ''.format(auc_sv),lw=2)

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
ax.grid(True)
plt.show()














