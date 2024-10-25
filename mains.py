#Import necessary libraries
#pip install xgboost
#pip install scikit-learn


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
import warnings
import os
from scipy.stats import f_oneway
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
import time



print("Program is runnig")
print()
start_time = time.time()




#load the datasets

a1 = pd.read_excel("case_study1.xlsx")
a2 = pd.read_excel("case_study2.xlsx")


df1 = a1.copy()
df2 = a2.copy()

df1.head(5)
df2.head(5)


df1.info()  ##Understanding the datasets
df2.info()

df1.describe()  ##statistical details
df2.describe()

df1.shape

##Removing Null Values as there is only 40 null values in first dataset 

df1 = df1.loc[df1['Age_Oldest_TL'] != -99999]
df1.shape


##As there is a big number of features that contains null values, I am imputing the columns having more than 10000 null values.

columns_to_be_removed = []

for i in df2.columns:
    if df2.loc[df2[i] == -99999].shape[0] > 10000:
        columns_to_be_removed.append(i)
        
        


df2 = df2.drop(columns_to_be_removed, axis = 1)


for i in df2.columns:
    df2 = df2.loc[df2[i] != -99999]
    
    
df2.isna().sum()

df1.isna().sum()



##Checking common columns in both the dataframe so that we can merge the datasets

for i in list(df1.columns):
    if i in list(df2.columns):
        print(i)
        
        
##Merging the dataframes with PROSPECTID column so that no nulls are present

df = pd.merge(df1, df2, how = 'inner', left_on = 'PROSPECTID', right_on = 'PROSPECTID')

df.head(5)
df.info()
df.isna().sum()


##Checking the number of categorical columns in df

for i in df.columns:
    if df[i].dtype == 'object':
        print(i)
        
        
print(df['MARITALSTATUS'].value_counts())
print(df['EDUCATION'].value_counts())
print(df['GENDER'].value_counts())
print(df['last_prod_enq2'].value_counts())
print(df['first_prod_enq2'].value_counts())
print(df['Approved_Flag'].value_counts())


##Contingency table with the help chi2 for Marital Status feature and Approved_flag 

#Chi-square test

for i in ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']:
    chi2, pval, _, _ = chi2_contingency(pd.crosstab(df[i], df['Approved_Flag']))
    
    print(i, '---', pval)
    


##Since all the categorical features have p-value <= 0.05 we will accept all.



##VIF for the numerical columns::
    

numeric_columns = []

for i in df.columns:
    if df[i].dtype != 'object' and i not in ['PROSPECTID','Approved_Flag']:
        numeric_columns.append(i)
        


##VIF Sequentially check

vif_data = df[numeric_columns]
total_columns = vif_data.shape[1]
columns_to_be_kept = []
column_index = 0

for i in range(0, total_columns):
    
    vif_value = variance_inflation_factor(vif_data, column_index)
    print(column_index, '----', vif_value)
    
    
    if vif_value <= 6:
        columns_to_be_kept.append(numeric_columns[i])
        column_index = column_index + 1
        
    else:
        vif_data = vif_data.drop([numeric_columns[i]], axis = 1)
        
        


##Check Anova test for columns_to_be_kept

columns_to_be_kept_numerical = []

for i in columns_to_be_kept:
    a = list(df[i])
    b = list(df['Approved_Flag'])
    
    
    group_P1 = [value for value, group in zip(a, b) if group == 'P1']
    group_P2 = [value for value, group in zip(a, b) if group == 'P2']
    group_P3 = [value for value, group in zip(a, b) if group == 'P3']
    group_P4 = [value for value, group in zip(a, b) if group == 'P4']
    
    
    f_statistic, p_value = f_oneway(group_P1, group_P2, group_P3, group_P4)
    
    
    if p_value <= 0.05:
        columns_to_be_kept_numerical.append(i)
        

        

df['EDUCATION'].unique()
        
        
        
##Label encoding for Education feature


df.loc[df['EDUCATION'] == 'SSC', ['EDUCATION']]                            = 1
df.loc[df['EDUCATION'] == '12TH', ['EDUCATION']]                           = 2
df.loc[df['EDUCATION'] == 'GRADUATE', ['EDUCATION']]                       = 3
df.loc[df['EDUCATION'] == 'UNDER GRADUATE', ['EDUCATION']]                 = 3
df.loc[df['EDUCATION'] == 'POST-GRADUATE', ['EDUCATION']]                  = 4
df.loc[df['EDUCATION'] == 'OTHERS', ['EDUCATION']]                         = 1
df.loc[df['EDUCATION'] == 'PROFESSIONAL', ['EDUCATION']]                   = 3



df['EDUCATION'].value_counts()
df['EDUCATION'] = df['EDUCATION'].astype(int)
df.info()       


df_encoded = pd.get_dummies(df, columns=['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']) 
        
 
        
df_encoded.info()





####Machine Learning model fitting

##Model training

##Data Processing

## Algorithm - 01 :::: Random Forest

y = df_encoded['Approved_Flag']
X = df_encoded.drop(['Approved_Flag'], axis = 1)


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state= 42)

rf_classifier = RandomForestClassifier(n_estimators = 200, random_state= 42)

rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict((X_test))




accuracy = accuracy_score(y_test, y_pred)
print()
print(f'Accuracy: {accuracy}')
print()

precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)


for i , v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class: {v}")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score : {f1_score[i]}")
    print()
    
    
    
    
    
    
### Algorithm 02 : XGBoost

xgb_classifier = xgb.XGBClassifier(objective= 'multi:softmax', num_class = 4)

##y = df.encoded['Approved_Flag']
##X = df_encoded.drop(['Approved_Flag'], axis = 1)


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size = 0.2, random_state = 42)


xgb_classifier.fit(X_train, y_train)

y_pred = xgb_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print()
print(f'Accuracy: {accuracy : .2f}')
print()

precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)


for i , v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class: {v}")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score : {f1_score[i]}")
    print()




##Decision Tree

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.6, random_state = 42)

dt_model = DecisionTreeClassifier(max_depth = 5, min_samples_split= 5)
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print()
print(f'Accuracy: {accuracy : .2f}')
print()

precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)


for i , v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class: {v}")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score : {f1_score[i]}")
    print()


from sklearn.metrics import confusion_matrix

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)




##Hyper Parameter tunig for XGBoost Model for optimisation

##Defining Hyper Parameter grids


param_grid = {
    'colsample_bytree' : [0.1, 0.3, 0.5, 0.7, 0.9],
    'learning_rate' : [0.001, 0.01, 0.1, 1],
    'max_depth' : [3, 5, 8, 10],
    'n_estimators' : [10, 50, 100],
    'alpha' : [1,10,100]
    
    }



index = 0


answer_grid = {
    'combinations'                 :[],
    'train_Accuracy'               :[],
    'test_Accuracy'                :[],
    'colsample_bytree'             :[],
    'learning_rate'                :[],
    'max_depth'                    :[],
    'alpha'                        :[],
    'n_estimators'                  :[]
    
    }




## Now we will loop through each combination of hyperparameters

for colsample_bytree in param_grid['colsample_bytree']:
    for learning_rate in param_grid['learning_rate']:
        for max_depth in param_grid['max_depth']:
            for alpha in param_grid['alpha']:
                for n_estimators in param_grid['n_estimators']:
                    
                    
                    
                    index = index+1
                    
                    
                    ##Define and train XGBoost model for all the combinations above we have created
                    
                    model = xgb.XGBClassifier(objective = 'multi:softmax',
                                              num_class = 4,
                                              colsample_bytree = colsample_bytree,
                                              learning_rate = learning_rate,
                                              max_depth = max_depth,
                                              alpha = alpha,
                                              n_estimators = n_estimators)
                    
                    
                    y = df_encoded['Approved_Flag']
                    X = df_encoded.drop(['Approved_Flag'], axis = 1)
                    
                    label_encoder = LabelEncoder()
                    
                    y_encoded = label_encoder.fit_transform(y)
                    
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size= 0.20, random_state=42)
                    
                    
                    model.fit(X_train, y_train)
                    
                    
                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)
                    
                    train_accuracy = accuracy_score(y_train,y_pred_train)
                    test_accuracy = accuracy_score(y_test, y_pred_test)
                    
                    
                    ##Add all the details in the list we have created to store information above
                    
                    
                    answer_grid['combinations'].append(index)
                    answer_grid['train_Accuracy'].append(train_accuracy)
                    answer_grid['test_Accuracy'].append(test_accuracy)
                    answer_grid['colsample_bytree'].append(colsample_bytree)
                    answer_grid['learning_rate'].append(learning_rate)
                    answer_grid['max_depth'].append(max_depth)
                    answer_grid['alpha'].append(alpha)
                    answer_grid['n_estimators'].append(n_estimators)
                    
                    print(f"combination {index}")
                    print(f"colsample_bytree {colsample_bytree}, learning_rate: {learning_rate}, max_depth: {max_depth}, alpha : {alpha}, n_estimators : {n_estimators}")
                    print(f"Train Accuracy : {train_accuracy:.2f}")
                    print(f"Test Accuracy : {test_accuracy:.2f}")
                    print("-" * 30)


##Save the model

import pickle
filename = 'crm_v1.sav'
pickle.dump(model, open(filename, 'rb'))


















                    
                    
                
                    
                    
                    































































        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        





































