# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 09:54:20 2024

@author: manas
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder, MinMaxScaler
import statsmodels.api as sm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report 
from sklearn.ensemble import RandomForestClassifier 
from xgboost import XGBClassifier  
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# CALLING DATA
data=pd.read_csv("D:/Documents/dissertation_msc/adult_clean.csv")
data=data[data['age']<=73.5]


# DIVIDING DATA
X=data.drop(columns=['income'])
y=data['income']


 #SEPARATING DATA

label_encoder=LabelEncoder()

y=label_encoder.fit_transform(y)

X=pd.get_dummies(X,columns=None,drop_first=True)



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)

# STANDARDISING DATA
scaler=StandardScaler()
X_train_sc=scaler.fit_transform(X_train)
X_test_sc=scaler.transform(X_test)


X_scaled_df = pd.DataFrame(X_train_sc, columns=X_train.columns)


from sklearn.linear_model import LogisticRegression
# Liblinear is a solver that is very fast for small datasets, like ours
model = LogisticRegression()
# Fit the model to the training data
model.fit(X_train_sc, y_train)

y_h_log=(model.predict(X_test_sc)).astype(int)
#y_h_log_train=(result.predict(X_train_sc) >= th_f['threshold']).astype(int)


def metrics_fun(y_test,y_hat):
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_hat, average='weighted')
    accuracy=accuracy_score(y_test, y_hat)
    
    # Print the results
    print("Precision (Weighted):", precision)
    print("Recall (Weighted):", recall)
    print("F1-Score (Weighted):", f1_score)
    print("accuracy (Weighted):", accuracy)

metrics_fun(y_test,y_h_log)


# DECISION TREE


params={
        'max_depth':[2,3,5,10,20],
        'min_samples_leaf':[5,10,20,50,100],
        'min_samples_split':[4,6,8,10,20],
        'criterion':['gini','entropy']}


dt1=DecisionTreeClassifier(random_state=1)

clf=GridSearchCV( estimator=dt1,
                 param_grid=params,
                 cv=4,n_jobs=-1,verbose=1,
                 scoring='accuracy')



clf.fit(X_train_sc, y_train)

dt_best=clf.best_estimator_
y_h_dt = dt_best.predict(X_test_sc)
metrics_fun(y_test,y_h_dt)

# RANDOM FOREST

param_grid = { 
    'n_estimators': [25, 50, 100],  
    'max_depth': [3, 6, 9], 
    'max_leaf_nodes': [3, 6, 9], 
} 


grid_search = GridSearchCV(RandomForestClassifier(), 
                           param_grid=param_grid) 
grid_search.fit(X_train_sc, y_train) 
rf_best=grid_search.best_estimator_
print(grid_search.best_estimator_) 

y_h_rf= rf_best.predict(X_test_sc)

metrics_fun(y_test,y_h_rf)

#_______________________________

from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.algorithms.preprocessing.reweighing import Reweighing

def aif_data(X,y):
    encoded_df=X.copy()
    
    encoded_df['income']=y
    
    BL = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=encoded_df,
        label_names=['income'],
        protected_attribute_names=['sex_Male'])
    return(BL)


BL=aif_data(X,y)
def aif_met(BL):
    privileged_groups = [{'sex_Male': 1}]
    unprivileged_groups = [{'sex_Male': 0}]
    metric_transf_train = BinaryLabelDatasetMetric(BL, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
    
    return([metric_transf_train.disparate_impact(),metric_transf_train.mean_difference()])


def aif_cl(BL,BL_p):
    classified_metric = ClassificationMetric(BL, BL_p, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)

    return([classified_metric.equal_opportunity_difference()])





print(BL.protected_attribute_names)


BL_train=aif_data(X_train,y_train)

aif_met(BL_train)

#aif360.sklearn.metrics.equal_opportunity_difference

aif_cl(aif_data(X_test,y_test),aif_data(X_test,y_h_log))

#_______________________________________________________________


BL=aif_data(X,y)
di = DisparateImpactRemover(repair_level = 1.0)
dataset_transf_train = di.fit_transform(BL)
transformed = dataset_transf_train.convert_to_dataframe()[0]

transformed.to_csv("D:/Documents/dissertation_msc/transformed.csv")
x_DI = transformed.drop(['income'], axis = 1)
y_DI = transformed['income']

X_train_DI,X_test_DI,y_train_DI,y_test_DI = train_test_split(x_DI, y_DI, test_size=0.3, random_state = 1)

scaler = StandardScaler()
X_train_DI_sc = scaler.fit_transform(X_train_DI)

X_test_DI_sc = scaler.transform(X_test_DI)

model_DI = LogisticRegression()
# Fit the model to the training data
model_DI.fit(X_train_DI_sc, y_train_DI)

y_h_log_DI=(model.predict(X_test_DI_sc)).astype(int)

BL_test_log=aif_data(X_test,y_h_log_DI)
aif_met(BL_test_log)


aif_met(aif_data(X_test_DI,y_test_DI))

metrics_fun(y_test_DI,y_h_log_DI)
#____________________________________________________________

privileged_groups = [{'sex_Male': 1}]
unprivileged_groups = [{'sex_Male': 0}]
RW = Reweighing(unprivileged_groups=unprivileged_groups,
               privileged_groups=privileged_groups)

RW.fit(BL)
dataset_transf_train = RW.transform(BL)


transformed = dataset_transf_train.convert_to_dataframe()[0]



weights=dataset_transf_train.instance_weights
np.unique(weights)

#_______________
f=len(data[data['sex']=='Female'])/len(data)
m=len(data[data['sex']=='Male'])/len(data)

l=len(data[data['income']=='<=50K'])/len(data)

g=len(data[data['income']=='>50K'])/len(data)




f_l=len(data[(data['sex']=='Female')&(data['income']=='<=50K')])/len(data)
f_g=len(data[(data['sex']=='Female')&(data['income']=='>50K')])/len(data)
m_l=len(data[(data['sex']=='Male')&(data['income']=='<=50K')])/len(data)
m_g=len(data[(data['sex']=='Male')&(data['income']=='>50K')])/len(data)

f*g/f_g
f*l/f_l
m*g/m_g
m*l/m_l
#_______________













X_train,X_test,y_train,y_test,weights_train,weights_test=train_test_split(X,y,weights,test_size=0.3,random_state=1)

# STANDARDISING DATA
scaler=StandardScaler()
X_train_sc=scaler.fit_transform(X_train)
X_test_sc=scaler.transform(X_test)

#LOGISTIC MODEL
model = LogisticRegression()
# Fit the model to the training data
model.fit(X_train_sc, y_train,sample_weight=weights_train)

y_h_log_rw=(model.predict(X_test_sc)).astype(int)
#y_h_log_train=(result.predict(X_train_sc) >= th_f['threshold']).astype(int)

metrics_fun(y_test,y_h_log_rw)



#original test data
aif_met(aif_data(X_test,y_test))

# for predicted but with transform
aif_met(aif_data(X_test,y_h_log_rw))
aif_cl(aif_data(X_test,y_test),aif_data(X_test,y_h_log_rw))


# for predicted but wihout transform
aif_met(aif_data(X_test,y_h_log))
aif_cl(aif_data(X_test,y_test),aif_data(X_test,y_h_log))







# DECISION TREE


params={
        'max_depth':[2,3,5,10,20],
        'min_samples_leaf':[5,10,20,50,100],
        'min_samples_split':[4,6,8,10,20],
        'criterion':['gini','entropy']
        }


dt1=DecisionTreeClassifier(random_state=1)

clf=GridSearchCV( estimator=dt1,
                 param_grid=params,
                 cv=4,n_jobs=-1,verbose=1,
                 scoring='accuracy')



clf.fit(X_train_sc, y_train,sample_weight=weights_train)

dt_best=clf.best_estimator_
y_h_dt_rw = dt_best.predict(X_test_sc)
metrics_fun(y_test,y_h_dt)

metrics_fun(y_test,y_h_dt_rw)


# for predicted but with transform
aif_met(aif_data(X_test,y_h_dt_rw))
aif_cl(aif_data(X_test,y_test),aif_data(X_test,y_h_dt_rw))

# for predicted but wihout transform
aif_met(aif_data(X_test,y_h_dt))
aif_cl(aif_data(X_test,y_test),aif_data(X_test,y_h_dt))


#RANDOM FOREST---------------------



param_grid = { 
    'n_estimators': [25, 50, 100],  
    'max_depth': [3, 6, 9], 
    'max_leaf_nodes': [3, 6, 9], 
} 


grid_search = GridSearchCV(RandomForestClassifier(), 
                           param_grid=param_grid) 

grid_search.fit(X_train_sc, y_train,sample_weight=weights_train) 
rf_best=grid_search.best_estimator_
print(grid_search.best_estimator_) 

y_h_rf_rw= rf_best.predict(X_test_sc)

metrics_fun(y_test,y_h_rf_rw)

# for predicted but with transform
aif_met(aif_data(X_test,y_h_rf_rw))
aif_cl(aif_data(X_test,y_test),aif_data(X_test,y_h_rf_rw))

# for predicted but wihout transform
aif_met(aif_data(X_test,y_h_rf))
aif_cl(aif_data(X_test,y_test),aif_data(X_test,y_h_rf))


#__________________________________________________________________


#LOGISTIC MODEL
model = LogisticRegression()
# Fit the model to the training data
model.fit(X_train_sc, y_train)

y_h_log_rw=(model.predict(X_test_sc)).astype(int)
#y_h_log_train=(result.predict(X_train_sc) >= th_f['threshold']).astype(int)

metrics_fun(y_test,y_h_log_rw)



#original test data
aif_met(aif_data(X_test,y_test))

# for predicted but with transform
aif_met(aif_data(X_test,y_h_log_rw))s
aif_cl(aif_data(X_test,y_test),aif_data(X_test,y_h_log_rw))


# for predicted but wihout transform
aif_met(aif_data(X_test,y_h_log))
aif_cl(aif_data(X_test,y_test),aif_data(X_test,y_h_log))







# DECISION TREE


params={
        'max_depth':[2,3,5,10,20],
        'min_samples_leaf':[5,10,20,50,100],
        'min_samples_split':[4,6,8,10,20],
        'criterion':['gini','entropy']
        }


dt1=DecisionTreeClassifier(random_state=1)

clf=GridSearchCV( estimator=dt1,
                 param_grid=params,
                 cv=4,n_jobs=-1,verbose=1,
                 scoring='accuracy')



clf.fit(X_train_sc, y_train,sample_weight=weights_train)

dt_best=clf.best_estimator_
y_h_dt_rw = dt_best.predict(X_test_sc)
metrics_fun(y_test,y_h_dt)

metrics_fun(y_test,y_h_dt_rw)


# for predicted but with transform
aif_met(aif_data(X_test,y_h_dt_rw))
aif_cl(aif_data(X_test,y_test),aif_data(X_test,y_h_dt_rw))

# for predicted but wihout transform
aif_met(aif_data(X_test,y_h_dt))
aif_cl(aif_data(X_test,y_test),aif_data(X_test,y_h_dt))


#RANDOM FOREST---------------------



param_grid = { 
    'n_estimators': [25, 50, 100],  
    'max_depth': [3, 6, 9], 
    'max_leaf_nodes': [3, 6, 9], 
} 


grid_search = GridSearchCV(RandomForestClassifier(), 
                           param_grid=param_grid) 

grid_search.fit(X_train_sc, y_train,sample_weight=weights_train) 
rf_best=grid_search.best_estimator_
print(grid_search.best_estimator_) 

y_h_rf_rw= rf_best.predict(X_test_sc)

metrics_fun(y_test,y_h_rf_rw)

# for predicted but with transform
aif_met(aif_data(X_test,y_h_rf_rw))
aif_cl(aif_data(X_test,y_test),aif_data(X_test,y_h_rf_rw))

# for predicted but wihout transform
aif_met(aif_data(X_test,y_h_rf))
aif_cl(aif_data(X_test,y_test),aif_data(X_test,y_h_rf))
