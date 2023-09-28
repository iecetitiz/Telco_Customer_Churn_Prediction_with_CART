# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 12:24:42 2023

@author: iecet
"""
# Importing libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


df = pd.read_csv("C:/datasets/machine_learning/Telco-Customer-Churn.csv") 

df.info()

#Handling missing values
#If errors = ‘coerce’, then invalid parsing " " will be set as NaN.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Rows that 'TotalCharges' column has empty values
empty_total_charges = df[df["TotalCharges"].isnull()]

#Converting "Churn" column into numeric type
df["Churn"] = df["Churn"].apply(lambda x : 1 if x == "Yes" else 0)

# New customers are the reason for these empty values
new_customers = df[df['tenure'] == 0]

# Replacing empty values with '0' in 'TotalCharges' column
df["TotalCharges"].replace(np.nan, 0, inplace = True)



# Observing null values
df.isnull().sum()

# Taking columns based on their categories
def grab_col_names(dataframe, cat_th=10,  car_th=20):
    
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < 10 and dataframe[col].dtypes in ["int64", "float64"]]

    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > 20 and str(dataframe[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]
    
    print(f"Observations: {dataframe.shape[0]}") # satır
    print(f"Variables: {dataframe.shape[1]}") # değişken
    print(f'cat_cols: {len(cat_cols)}') # kategorik degişken
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

# Taking columns based on their categories
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Takes a summary of categorical columns
def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")


for col in cat_cols:
    cat_summary(df, col)
    
#Handling outliers
    
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
   quartile1 = dataframe[col_name].quantile(q1)
   quartile3 = dataframe[col_name].quantile(q3)
   interquantile_range = quartile3 - quartile1
   up_limit = quartile3 + 1.5 * interquantile_range
   low_limit = quartile1 - 1.5 * interquantile_range
   return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
    
for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)
        
#Removing the "Churn" column from cat_cols    
cat_cols = [col for col in cat_cols if col not in ["Churn"]]

# One Hot encoding
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

#Creating dff dataframe with encoded columns
dff = pd.get_dummies(df, columns=cat_cols, drop_first=True)


# Preparing data: Feature columns: X and a target column y
y = dff["Churn"]
X = dff.drop(["Churn","customerID"], axis=1)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the CART classifier
cart_classifier = DecisionTreeClassifier(random_state=42)

# Training the classifier on the training data
cart_classifier.fit(X_train, y_train)

# Using the trained model to make predictions on the test data
y_pred = cart_classifier.predict(X_test)

# Evaluating the model:
# Accuracy:
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}") #0.71

#Classification Report (provides precision, recall, F1-score, and support for each class):
print(classification_report(y_test, y_pred))

#       precision    recall  f1-score   

#0       0.81      0.80      0.81      
#1       0.46      0.47      0.46 


# AUC score
y_prob = cart_classifier.predict_proba(X_test)[:, 1]
roc_auc_score(y_test, y_prob)

# Cross Validation Scores

cv_results = cross_validate(cart_classifier,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_accuracy = cv_results["test_accuracy"]
cv_f1 = cv_results["test_f1"]

# Predicting for train set
train_pred = cart_classifier.predict(X_train)
print(classification_report(y_train, train_pred))

# Followings are the results of the predictions for train set, it's clear that there is an overfitting issue
#              precision    recall  f1-score   

       #0       1.00      1.00      1.00      
       #1       1.00      1.00      1.00      

#accuracy                           1.00 


################################################
# Hyperparameter Optimization with GridSearchCV
################################################

#Taking parameters for hyperparameter optimization
cart_classifier.get_params()

#Defining the hyperparameter grid to search over
param_grid = {
    'max_depth': range(1, 11),
    'min_samples_split': range(2, 20),
}

# Creating the GridSearchCV object and fitting the GridSearchCV object to train data:
cart_best_grid = GridSearchCV(cart_classifier,
                              param_grid, 
                              cv = 5,
                              n_jobs = -1,
                              verbose=True
                              ).fit(X, y)

# Getting the best hyperparameters
best_params = cart_best_grid.best_params_

# max_depth: 6
#min_samples_split: 8

################################################
# Final Model
################################################


cart_final = cart_classifier.set_params(**cart_best_grid.best_params_).fit(X, y)

cv_results = cross_validate(cart_final,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])


acc = cv_results['test_accuracy']

f1 = cv_results['test_f1']

roc_auc= cv_results['test_roc_auc']



# It can be seen that the precision,recall and f1-score values have increased significantly after optimization
y_predicto = cart_classifier.predict(X_test)
print(classification_report(y_test, y_predicto))


#              precision    recall  f1-score   

       #0       0.86      0.91      0.88      
       #1       0.70      0.59      0.64       

#accuracy                           0.83















