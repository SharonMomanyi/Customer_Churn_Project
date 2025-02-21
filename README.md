# Customer_Churn_Project

## OVERVIEW

Customer Churn is a major challange for businesses. In the Telecommunication industry, it occurs when customers switch to different providers due to pricing, service quality, or even competitive offers.

The aim is to understand the patterns of these customers who churn and also build a model that will predict whether a customer will 'soon' stop doing business with SyriaTel Telecommunications.

The stakeholders for this project are different departments in SyriaTel. These include the Executive Office, the Customer Relations and Support department and the Marketing Department.


**OBJECTIVES**

1. Identify the factors that contribute to customer churn.
2. Build a model to predict customer churn.


### Required Libraries
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler scaler = StandardScaler()
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


## DATA UNDERSTANDING 

We will be working with a dataset from Kaggle(https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset/code). The data is contained in `Customer_Churn_Data.csv` and each record represents a customer, and the columns are the attributes of the customers, ranging from their phone number, state, number of customer service calls they have made, whether they have left or are still a customer, among other attributes.


Here's a summary of the columns:

* state:  The state of the customer.

* account length:  The length of the account in days or months.

* area code:  The area code of the customer's phone number.

* phone number:  The phone number of the customer.

* international plan:  Whether the customer has an international plan or not.

* voice mail plan:  Whether the customer has a voicemail plan or not.

* number vmail messages:  The number of voicemail messages the customer has.

* total day minutes:  Total minutes of day calls.

* total day calls:  Total number of day calls.

* total day charge:  Total charge for the day calls.

* total eve minutes:  Total minutes of evening calls.

* total eve calls:  Total number of evening calls.

* total eve charge:  Total charge for the evening calls.

* total night minutes:  Total minutes of night calls.

* total night calls:  Total number of night calls.

* total night charge:  Total charge for the night calls.

* total intl minutes:  Total minutes of international calls.

* total intl calls:  Total number of international calls.

* total intl charge:  Total charge for the international calls.

* customer service calls:  Number of times the customer called customer service.

* churn:  Whether the customer churned or not (True/False).

#Loading and displaying the dataset
df = pd.read_csv("Customer_Churn_Data.csv")
df.head(10)

#Rename the column names
df.columns = df.columns.str.replace(" ", "_")
print(df.columns)

## EXPLORATORY DATA ANALYSIS

#Check the data types of the columns
df.info()

* We have object, integer, float and boolean data types. For modelling purposes, I will convert and transform relevant columns.


* There are no misssing values in the columns as seen above.

#Check descriptive statistics
df.describe()

## Univariate Analysis

This analysis is for exploring the distribution of some features for a better understanding.

**1. The target variable `'churn'`**

![Churn_distribution](Images/Churn_Distribution.PNG)

* Most customers fall under the `False` category. Those who churn range between 500-600 while those who dont are well over 2,500.

********************

**2. Which states have more customers?**

![Churn_distribution](Images/State_Distribution.PNG)

* The above plot shows the distribution of customers across the top 20 states with the most customers.

* West Virginia (WV) has the highest number of customers, followed by Minnesota (MN) then New York (NY). West Virginia has a significantlly higher number of customers compared to the other states.

***********************


## Bivariate Analysis

Let's now explore the relationship between some features and the target to look out for patterns and insights.

**1. International plan and Churn**

![Churn_distribution](Images/Churn_vs_International_Plan.PNG)

**FINDINGS**

* Customers with no international plan and are retained as customers are the most.

* Those who have an international plan and churn are less than those who dont churn but still have an internatonal plan.

* This shows that having an internationl plan is not a factor causing churning.

**2. Voice Mail Plan and Churn**

![Churn_distribution](Images/Voice_mail_plan_vs_Churn.PNG)

**FINDINGS**

* Same as with the international plan, most customers who are retained do not have a voice mail plan.

* For those with a voice mail plan, those who churn are still less than those who dont. This also shows that having a voicemail plan is really not a factor causing churning.

**3. Charges and Churn**

![Churn_distribution](Images/Average_Charge_by_Time_of_Day.PNG)

**FINDINGS**

* Charges are significantlly higher during the day compared to the evening and the night time. It is cheaper to use this service at night.

**4. Total Day Minutes and Churn**

![Churn_distribution](Images/Total_day_minutes_vs_Churn.PNG)

**FINDINGS**

* Higher churn rates are involved with customers who have a lot of daily minutes. Majority of those who are retained have about 140 - 210 day minutes while majority of those who churn have 150 - 260 day minutes.


**5. Total Night Minutes and Churn**

![Churn_distribution](Images/Total_Night_Minutes_vs_Churn.PNG)

**FINDINGS**

* Majority of the customers who churn have almost the same night minutes as majority of those who are retained.

* Comparing the churn rates at night and during the day using minutes, it is evident that most customers who churn have a lot of daily minutes, and therefore are the most expensed since charges are higher during the day.


**6. Customer Service Calls and Churn**

![Churn_distribution](Images/Customer_service_calls_vs_Churn.PNG)

**FINDINGS**

* Most customers who call customer service have churned compared to those who haven't churned.


### OVERALL FINDINGS AND RECOMMENDATIONS

**FINDINGS**

1. International plans and voicemail plans dont play a significant role in the customer churn rate.

2. High daily charges and poor customer service are leading to high customer churn rates.

**RECOMMENDATIONS**

1. Reduce the daily charges to be able to retain customers. The marketing team should come up with campaigns that will inform customers when this happens.

2. Train the Customer Relations and Support department on client management.


# DATA PREPARATION


#Define X and y
y = df[['churn']]
X = df.drop(['churn','phone_number'], axis=1)

#Split the data

Data preparation should happen after the data has been split to avoid data leakage.
Data leakage will lead to overly optimistic perfomance metrics during model validation since the model may have had access to information before the split.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train.head()

X_test.head()

## OneHotEncode categorical features


#Define categorical columns
cat_columns = ['state', 'area_code', 'international_plan', 'voice_mail_plan']

#instantiate OneHotEncoder
ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

#Fit and transform training data then convert to a dataframe
X_train_ohe = pd.DataFrame(ohe.fit_transform(X_train[cat_columns]), 
                           columns=ohe.get_feature_names_out(cat_columns), 
                           index=X_train.index)

#Transform Test set
X_test_ohe = pd.DataFrame(ohe.transform(X_test[cat_columns]),
                           index=X_test.index,
                           columns=ohe.get_feature_names_out(cat_columns))

#Drop original categorical columns from X_train and X_test
X_train = X_train.drop(columns=cat_columns)
X_test = X_test.drop(columns=cat_columns)

#Concatenate one-hot encoded features back
X_train = pd.concat([X_train, X_train_ohe], axis=1)
X_test = pd.concat([X_test, X_test_ohe], axis=1)

X_test = X_test[X_train.columns]

X_train.head()

X_test.head()

## Standardize Numeric Features

#Instantiate the scaler 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_numeric= X_train.copy()

#Filter to remain with relevant columns
numeric_cols = df.select_dtypes('number')
numeric_cols = numeric_cols.drop('area_code', axis=1)

X_train_numeric = X_train_numeric[numeric_cols.columns] # a subset of X_train, untransformed

#Standardize the training set
X_train_standardized = pd.DataFrame(scaler.fit_transform(X_train_numeric),
                                   columns=X_train_numeric.columns,
                                   index=X_train_numeric.index)
X_train_standardized.head()

#Transform the test set
X_test_standardized = pd.DataFrame(scaler.transform(X_test[numeric_cols.columns]),
                                   columns=X_test[numeric_cols.columns].columns,
                                   index=X_test[numeric_cols.columns].index)
X_test_standardized.head()

### Combining the datasets into one

#Drop non-transformed columns
cols_to_drop = cat_columns + numeric_cols.columns.tolist()
X_train_filtered = X_train.drop(columns=cols_to_drop, errors='ignore')

#Bring them together
X_train_tr = pd.concat([X_train_filtered, X_train_ohe, X_train_standardized], axis=1)
X_train_tr.head()

X_train_tr.head(10)

X_train_tr.info()

The `'phone_number'` column seems to be a unique identifier for the customers so I am chooseing to drop it.

### Combining Test set into one.

#Drop non-transformed columns
cols_to_drop = cat_columns + numeric_cols.columns.tolist()
X_test_filtered = X_test.drop(columns=cols_to_drop, errors='ignore')

#Bring them together
X_test_tr = pd.concat([X_test_filtered, X_test_ohe, X_test_standardized], axis=1)
X_test_tr.head()

# MODELLING

## 1. Baseline Model

Let's build a regression model, which will help classify whether a customer will leave, `True` or not, `False`

logreg = LogisticRegression(fit_intercept=False, C=1e12, solver='liblinear')
baseline_model = logreg.fit(X_train_tr, y_train)
baseline_model

### Model Evaluation

y_pred_train = baseline_model.predict(X_train_tr)
y_pred_test = baseline_model.predict(X_test_tr)

from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
print("Training Set Accuracy:", accuracy_score(y_train, y_pred_train))
print("Testing Set Accuracy:", accuracy_score(y_test, y_pred_test))
print()
print("Training Set Confusion Matrix:\n", confusion_matrix(y_train, y_pred_train))
print("Testing Set Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))
print()
print("Training Set Classification Report:\n", classification_report(y_train, y_pred_train))
print("Testing Set Classification Report:\n", classification_report(y_test, y_pred_test))

### Interpretation

**1. ACCURACY SCORE**

Training Set Accuracy: 0.872

Testing Set Accuracy: 0.858


The accuracy score for the training test is 87.2% while that for the test set is 85.8%. There is a slight difference meaning the model generalizes well and there is no overfitting.

**2. CONFUSION MATRIX**

`Training Set Confusion Matrix:`

                              [[1932   61]

                              [ 238  102]]
 
* True Negatives(TN) = 1932
The model correctly predicted **1932** negative cases.


* False Positives(FP) = 61 
The model predicted **61** cases as positive, when infact they are negative.


* False Negatives(FN) = 238 
The model has predicted **238** cases as negative when they are actually positive


* True Positives(TN) = 102
The model has correctly predicted **102** positive cases.
 
 
 `Testing Set Confusion Matrix:`
 
                               [[826  31]
                               
                               [111  32]]
                               
* True Negatives(TN) = 826
The model correctly predicted **826** negative cases.


* False Positives(FP) = 31 
The model predicted **31** cases as positive, when infact they are negative.


* False Negatives(FN) = 111 
The model has predicted **111** cases as negative when they are actually positive


* True Positives(TN) = 32
The model has correctly predicted **32** positive cases.

**************************
From both the training and testing confusion matrices, it is evident that the model is better at predicting False, that a customer wont leave than at predicting True, that's when they will leave.

The False negatives are high for both sets and this is an issue since the model is classifying customers that will leave as customers that will stay.

**3.RECALL, PRECISION AND F1-SCORE**

`Training Set Classification Report:`

               precision    recall  f1-score

       False       0.89      0.97      0.93     
        True       0.63      0.30      0.41      



`Testing Set Classification Report:`

               precision    recall  f1-score   

       False       0.88      0.96      0.92       
        True       0.51      0.22      0.31       

**TEST INTERPRETATION**
* The precision for `False` predictions is at  88% meaning that the model is predicting a high percentage of false cases correctly.


* For recall, the true positive rate, the model is failing to identify many positive instances, evidenced by the low value of 22%.

* An F1-score of 31% for the `True` class show that the model is not mininmizing the false negatives and false positives efficiently as this is a low percentage compared to 92% for the False class. 

*******************************
From the above analysis, it is evident that the dataset is imbalnced as we have more negatives than positives.

## 2. Random Forest 

We will now build a Random Forest classifier since Logistic regression is producing imbalanced classes.

The parameters we will include in the model are:
        `class_weight='balanced'` to deal with the class imbalance,
        `n_estimators=100` which is the number of decision trees in the forest and
        `random_state=42` which ensures reproducibility

#Initialize the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

#Train the model
rf_model.fit(X_train_tr, y_train)

### Model Evaluation

#Make predictions
y_train_pred_rf = rf_model.predict(X_train_tr)
y_test_pred_rf = rf_model.predict(X_test_tr)

print("Training Accuracy:", accuracy_score(y_train, y_train_pred_rf))
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred_rf))
print()
print("Training Confusion Matrix:\n", confusion_matrix(y_train, y_train_pred_rf))
print("Testing Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred_rf))
print()
print("Classification Report:\n", classification_report(y_train, y_train_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_test_pred_rf))

## Interpretation

**1. ACCURACY**

* The 100% accuracy score on the training data suggests overfitting.

* In comparison to the accuracy score from the baseline model, this model generalizes better to unseen data  since it has a higher score, of 93.1%

**2. CONFUSION MATRIX**

`Training Set Confusion Matrix:`

                         [[1993    0]
 
                         [   0  340]]
                     
* There are **1993** True Negatives and **340** True Positives, which is all the data we have. This shows that the model is overfitting. 


`Testing Set Confusion Matrix:`

                         [[855   2]
                     
                         [ 67  76]]

* True Negatives(TN)

  The model correctly predicted **855** negative cases.


* False Positives(FP)

  The model predicted **2** cases as positive, when infact they are negative.


* False Negatives(FN)  

  The model has predicted **67** cases as negative when they are actually positive


* True Positives(TN)

  The model has correctly predicted **76** positive cases.

**********

In comparison to the baseline model, this model has reduced the number of false positive cases to 2 from 31, meaning the prediction of customers who leave but its reported they are still customers has reduced. This is good.

We still have 67 cases of customers who leave being reported as staying. It has reduced compared to the baseline model, but is stll a worry-some number. The goal is to minimize it.

**3.RECALL, PRECISION AND F1-SCORE**

`Training Set Classification Report:`

               precision    recall  f1-score

       False       1.00      1.00      1.00     
        True       1.00      1.00      1.00      

* The model is overfitting on the training set as all scores are 1.00

`Testing Set Classification Report:`

               precision    recall  f1-score   

       False       0.93      1.00      0.96       
        True       0.97      0.53      0.69

* In the testing set, the precision and recall for the True class have improved in comparison to the baseline model.

The recall for the true class is at 53%, but it could be better.

## Hyperparameter Tuning

Since the model is overfitting on the training data, i will hypertune the parameters to reduce it.

We will use `GridSearchCV` to find the best parameters.

#Import libraries
from sklearn.model_selection import GridSearchCV

#Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'class_weight': ['balanced'] 
}

#Use GridSearchCV for tuning
grid_search = GridSearchCV(rf_model, param_grid, cv=3, scoring='recall', n_jobs=-1, verbose=2)

#Fit on training data
grid_search.fit(X_train_tr, y_train)

#Best parameters
print("Best Parameters:", grid_search.best_params_)

### Model Evaluation

Now to train the model with the best parameters then evaluate its perfomance.

#Training the best model
best_rf = grid_search.best_estimator_

#Make predictions
y_train_pred_best = best_rf.predict(X_train_tr)
y_test_pred_best = best_rf.predict(X_test_tr)

#Evaluate Performance
print("Training Accuracy:", accuracy_score(y_train, y_train_pred_best))
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred_best))
print()
print("Testing Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred_best))
print()
print("Testing Classification Report:\n", classification_report(y_test, y_test_pred_best))

## Interpretation
**1. ACCURACY**

* The overfitting in the train model has reduced. This is because the score has reduced from 100% to 95.1%.

* The Test accuracy of 93.8% is better compared to the baseline logistic regression 85.8% and untuned Random Forest of 93.1%. This model generalizes better compared to the other twomodels.

**2. CONFUSION MATRIX**

`Testing Set Confusion Matrix:`

                         [[822   35]
                     
                         [ 27  116]]

* True Negatives(TN) = 822
The model has correctly predicted **822** negative cases.


* False Positives(FP) = 32 
The model has predicted **32** cases as positive, when infact they are negative.


* False Negatives(FN) = 27 
The model has predicted **27** cases as negative when they are actually positive


* True Positives(TN) = 116
The model has correctly predicted **116** positive cases.


Initially the model predicted 67 False Negatives, and i was aiming to reduce this number so that the number of customers who leave but are reported as still customers reduces.

The number of True Positives has also increased, from 76 to 116, implying the True Positive Rate(TPR) has improved.


**3.RECALL, PRECISION AND F1-SCORE**

`Testing Set Classification Report:`

               precision    recall  f1-score   

       False       0.97      0.96      0.96       
        True       0.77      0.81      0.79
        
        
 * This model detects the True class much better than the untuned model, and this is evident from the recall score which has increased to 81% from 53%. 
 
## Feature Importance
feature_importance = best_rf.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X_train_tr.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
feature_importance_df.head(10)

For this model, these are the features that heavily impact the predictions of a customer churning or not.
The top 3 features contributing the most to customer churn are in agreement with the findings from the Exploratory Data Analysis.

 ## 3. Decision Tree 
 
#Initialize the model
dt_model = DecisionTreeClassifier(criterion='entropy', random_state=42)

#Train the model
dt_model.fit(X_train_tr, y_train)

## Model Evaluation

#Make predictions
y_train_pred_dt = dt_model.predict(X_train_tr)
y_test_pred_dt = dt_model.predict(X_test_tr)

print("Training Accuracy:", accuracy_score(y_train, y_train_pred_dt))
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred_dt))
print()
print("Training Confusion Matrix:\n", confusion_matrix(y_train, y_train_pred_dt))
print("Testing Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred_dt))
print()
print("Training Set Classification Report:\n", classification_report(y_train, y_train_pred_dt))
print("Testing Set Classification Report:\n", classification_report(y_test, y_test_pred_dt))

### Interpretation

**1. ACCURACY SCORE**

Training Set Accuracy: 1.00

Testing Set Accuracy: 0.91


The accuracy score for the training test is 100% while that for the test set is 91%. The 100% shows that there is overfitting since the model perfectly predicts the training data.

**2. CONFUSION MATRIX**

`Training Set Confusion Matrix:`

                              [[1933   0]

                              [ 0    340]]


* There are 1993 True Negatives and 340 True Positives, which is all the data we have. This shows that the model is overfitting.
 
 
 `Testing Set Confusion Matrix:`
 
                               [[808  49]
                               
                               [43   100]]
                               
* True Negatives(TN) = 808
The model correctly predicted **808** negative cases.


* False Positives(FP) = 49 
The model predicted **49** cases as positive, when infact they are negative.


* False Negatives(FN) = 43 
The model has predicted **43** cases as negative when they are actually positive


* True Positives(TN) = 100
The model has correctly predicted **100** positive cases.

**************************
This model is missing 43 customers that churn (False Negatives) and is predicting that 49 retained customers churn (False Positives).


**3.RECALL, PRECISION AND F1-SCORE**

`Training Set Classification Report:`

               precision    recall  f1-score

       False       1.00      1.00      1.00     
        True       1.00      1.00      1.00     



`Testing Set Classification Report:`

               precision    recall  f1-score   

       False       0.95      0.94      0.95       
        True       0.67      0.70      0.68       


* The precision for `False` predictions is at  95% meaning that the model is predicting quite a high percentage of false cases correctly.

* For the `True` class, its predicting 67% of True cases correctly. This is evidence for an imbalanced prediction, comparaed to the false cases. 


* For recall, the true positive rate, the model is correctly predicting 70% of the `True` class and 94% of the `False` class showing an imbalance as well.


* An F1-score of 68% for the `True` class compared to 95% for the `False` class show that the model is not mininmizing the false negatives and false positives for the `True` class as good as it is for the `False` class.

## Hypertuning Parameters

To deal with the overfittig, let's hypertune the parameters. We will use `GridSearchCV` to find the best parameters.

from sklearn.model_selection import GridSearchCV

#Define the parameter grid
param_grid_dt = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

#Initialize Grid Search
#grid_search_dt = GridSearchCV(estimator=dt_model, param_grid=param_grid_dt, cv=5, scoring='recall')
grid_search_dt = GridSearchCV(dt_model, param_grid_dt, cv=5, scoring='recall')

#Fit on training data
grid_search_dt.fit(X_train_tr, y_train)

#Best parameters
print("Best Parameters:", grid_search_dt.best_params_)

## Model Evaluation

#Train the best model
best_dt = grid_search_dt.best_estimator_

#Make predictions
y_train_pred_best_dt = best_dt.predict(X_train_tr)
y_test_pred_best_dt = best_dt.predict(X_test_tr)

#Evaluate Performance
print("Training Accuracy:", accuracy_score(y_train, y_train_pred_best_dt))
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred_best_dt))
print()
print("Testing Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred_best_dt))
print()
print("Testing Classification Report:\n", classification_report(y_test, y_test_pred_best_dt))

### Interpretation

**1. ACCURACY**
* This model has good accuracy scores for both the training and testing sets. There is no overfitting. 

**2. CONFUSION MATRIX**

* True Negatives(TN)

  The model correctly predicted 838 negative cases.


* False Positives(FP) 

  The model predicted 19 cases as positive, when infact they are negative.


* False Negatives(FN)

  The model has predicted 48 cases as negative when they are actually positive


* True Positives(TN) 

  The model has correctly predicted 95 positive cases.
  
**3.RECALL, PRECISION AND F1-SCORE**

* The precision for `False` predictions is at 95% while that for the `True` class is at 83%. There is a class imbalance.

* The model is correctly predicting 98% of the `False` class and 68% of the `True` class. More evidence of imbalanceness.

* An F1-score of 74% for the True class compared to 96% for the False class show that the model is not mininmizing the false negatives and false positives for the True class as good as it is for the False class.

*************************
It is very evident that this model is quite imbalanced and needs techniques like **SMOTE** which will generate synthetic samples for the `True` class which is the minority so as to balance the dataset.



## MODEL COMPARISON.

* The reason for building this models is to be able to predict whether a customer will stop doing business with SyriaTel. 

* The best model should therefore have a great recall score, ie. it should be able to detect True cases well, whilst reducing overfitting on the training data. 

* The Hypertuned Random Forest Model is doing just that compared to the Logistric Regression Model and the Decision Tree Model. 

## MODEL USEFULNESS

* The Tuned RandomForest Model should be used to predict customers who churn. The company should come up with personalised offers for them to try and retain them.

