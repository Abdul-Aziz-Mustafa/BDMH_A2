# Importing necessary libraries
import pandas as pd
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings('ignore')
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasClassifier
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report, confusion_matrix 

# Define hyperparameters to tune using GridSearchCV
params = {
    'max_features': [1, 3, 10, 12 ,20, 30, 50],
    'min_samples_split': [2, 3, 10, 20],
    'min_samples_leaf': [1, 3, 10,15],
    'bootstrap': [False],
    'n_estimators' :[100,300,500],
    'criterion': ["entropy", "gini"]
}

# Create a cross-validation object for hyperparameter tuning
cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)

# Load the training data
df_train = pd.read_csv(str(sys.argv[1]))
# Drop rows with missing values
df_train = df_train.dropna()
# Get column names
col = df_train.columns[:-1]
print(col)
# Split data into features and target variable
X = df_train.drop(['label'], axis=1)
y= df_train['label']

# Create a SelectKBest object for feature selection
# fs = SelectKBest(score_func=f_regression, k=400)
# # Apply feature selection
# X = fs.fit_transform(X, y)
# filter = fs.get_support()
# print(col[filter])

# Create a random forest classifier object
rfc = RandomForestClassifier()

# Perform hyperparameter tuning using GridSearchCV
grid = GridSearchCV(rfc, param_grid=params, scoring='accuracy', n_jobs =-1, cv=cv, verbose=1)
grid.fit(X, y)

# Print the best hyperparameters found
print('Best Score:', grid.best_score_)
print('Best Params:', grid.best_params_)
print('Best Estimator:', grid.best_estimator_)

# Use the best hyperparameters to create a random forest classifier
# classification = RandomForestClassifier(bootstrap=False, max_features=1, min_samples_split=3, n_estimators=300)
classification = RandomForestClassifier(bootstrap=False, max_features=1, n_estimators=500)

# Train the random forest classifier on the training data
classification.fit(X, y)

# Load the test data
df_test = pd.read_csv(str(sys.argv[2]))
# Extract features
X_test = df_test

# Make predictions on the test data using the trained classifier
df_result = pd.DataFrame(classification.predict_proba(X_test)[:,1])

# Generate IDs for each prediction
id = []
for i in range(len(df_result)):
    id.append(10000+i+1)
df_result['ID'] = id
df_result.columns = ['Label','ID']
df_result = df_result[['ID','Label']]

# Save the results to a CSV file
df_result.to_csv('5_output.csv', index=None)
df_result

