# Telco_Customer_Churn_Prediction_with_CART

Welcome to the "Telco Customer Churn Prediction" repository, where we use Classification and Regression Trees (CART) models to anticipate customer attrition in the telecommunications industry. 

The data set for this classification problem is taken from Kaggle. You can follow this [link](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)to download the dataset.

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


