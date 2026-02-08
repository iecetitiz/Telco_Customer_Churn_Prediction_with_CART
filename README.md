# Telco Customer Churn Prediction using CART

This project focuses on predicting customer churn for a telecommunications company using a **Decision Tree (Classification and Regression Tree - CART)** model. The analysis involves a complete end-to-end machine learning pipeline, from data cleaning and feature engineering to hyperparameter optimization.

## 📋 Project Overview
Customer churn is a critical metric for service-based industries. By identifying patterns in customer behavior (such as tenure, contract type, and monthly charges), this model helps predict which customers are likely to leave, enabling the business to implement retention strategies.

The data set for this classification problem is taken from Kaggle. You can follow this [link](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) to download the dataset.

Also following feature informations are taken from [Towards Data Science](https://towardsdatascience.com/machine-learning-case-study-telco-customer-churn-prediction-bc4be03c9e1d) 

## Meaning of Features
By inspecting the columns and their unique values, a general understanding about the features can be build. The features can also be clustered into different categories:

### Classification labels

Churn — Whether the customer churned or not (Yes or No)
### Customer services booked

* PhoneService — Whether the customer has a phone service (Yes, No)
* MultipleLines — Whether the customer has multiple lines (Yes, No, No phone service)
* InternetService — Customer’s internet service provider (DSL, Fiber optic, No)
* OnlineSecurity — Whether the customer has online security (Yes, No, No internet service)
* OnlineBackup — Whether the customer has online backup (Yes, No, No internet service)
* DeviceProtection — Whether the customer has device protection (Yes, No, No internet service)
* TechSupport — Whether the customer has tech support (Yes, No, No internet service)
* StreamingTV — Whether the customer has streaming TV (Yes, No, No internet service)
* StreamingMovies — Whether the customer has streaming movies (Yes, No, No internet service)
### Customer account information

* Tenure — Number of months the customer has stayed with the company
* Contract — The contract term of the customer (Month-to-month, One year, Two year)
* PaperlessBilling — Whether the customer has paperless billing (Yes, No)
* PaymentMethod — The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
* MonthlyCharges — The amount charged to the customer monthly
* TotalCharges — The total amount charged to the customer
### Customers demographic information

* customerID — Customer ID
* Gender — Whether the customer is a male or a female
* SeniorCitizen — Whether the customer is a senior citizen or not (1, 0)
* Partner — Whether the customer has a partner or not (Yes, No)
* Dependents — Whether the customer has dependents or not (Yes, No)

  

## 🛠 Features & Workflow

### 1. Data Cleaning & Preprocessing
* **Missing Value Handling:** Converted `TotalCharges` to numeric and filled missing values (linked to new customers with 0 tenure) with 0.
* **Target Encoding:** Converted the `Churn` status from "Yes/No" to binary 1/0.
* **Outlier Suppression:** Applied specialized functions to detect and cap outliers using the **Interquartile Range (IQR)** method at the 5th and 95th percentiles.

### 2. Exploratory Data Analysis (EDA)
* Developed a robust `grab_col_names` function to automatically categorize features into **Categorical**, **Numerical**, or **Cardinal** types.
* Generated automated summary reports for categorical variables to visualize class distributions.

### 3. Feature Engineering
* **One-Hot Encoding:** Prepared categorical variables for the model using `pd.get_dummies`, ensuring the model can interpret non-numeric data.

### 4. Hyperparameter Optimization
* Addressed **overfitting** (where the initial model had 100% accuracy on training data but performed poorly on test data).
* Utilized `GridSearchCV` to find the best values for:
    * `max_depth`
    * `min_samples_split`

---

## 📊 Results

After tuning the hyperparameters, the model showed a significant improvement in its ability to generalize to new data.

| Metric | Base Model (Test) | Final Model (Optimized) |
| :--- | :---: | :---: |
| **Accuracy** | 0.71 | **0.83** |
| **Precision (Churn: 1)** | 0.46 | **0.70** |
| **Recall (Churn: 1)** | 0.47 | **0.59** |
| **F1-Score (Churn: 1)** | 0.46 | **0.64** |

---



