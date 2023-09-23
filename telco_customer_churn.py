# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 12:24:42 2023

@author: iecet
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score


df = pd.read_csv("C:/datasets/machine_learning/Telco-Customer-Churn.csv") 

df.info()

# 'TotalCharges' column has empty values
empty_total_charges = df[df['TotalCharges'] == ' ']

# New customers are the reason for these empty values
new_customers = df[df['tenure'] == 0]

# Replacing empty values with '0' in 'TotalCharges' column
df['TotalCharges'].replace(' ', 0, inplace=True)

# 'TotalCharges' column is object type but actual type is float
df['TotalCharges'] = df['TotalCharges'].astype(float)


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
    
# Scaling numeric columns
standart_sclaler = StandardScaler()
df[num_cols] =standart_sclaler.fit_transform(df[num_cols]) 

# One Hot encoding
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)


# Preparing data: Feature columns: X and a target column y
X = df_encoded.drop(["customerID", "Churn_Yes"], axis = 1)
y = df_encoded["Churn_Yes"]

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

#precision    recall  f1-score   

#0       0.81      0.80      0.80      
#1       0.45      0.47      0.46 


# AUC score
y_prob = cart_classifier.predict_proba(X_test)[:, 1]
roc_auc_score(y_test, y_prob)



# Predicting 
train_pred = cart_classifier.predict(X_train)
print(classification_report(y_train, train_pred))

# Following are the predictions for train set, it's clear that there is an overfitting issue
#              precision    recall  f1-score   

       #0       1.00      1.00      1.00      
       #1       1.00      1.00      1.00      

#accuracy                           1.00 

# Cross Validation Scores

cv_results = cross_validate(cart_classifier,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_accuracy = cv_results["test_accuracy"]
cv_f1 = cv_results["test_f1"]












