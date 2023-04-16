Group-5
Members :
Abdul Aziz (2020485)
Manav Saini(2020518)
Manik Arora(2020519)

The Python script provided implements a machine learning model using the Random Forest Classifier (RFC) algorithm. First, the required libraries for data manipulation and visualization are imported, as well as the RFC model from the scikit-learn package. The code then defines a set of parameters to be tested using GridSearchCV, a tool for tuning hyperparameters of machine learning models. The parameters include the maximum number of features, minimum number of samples required to split a node, minimum number of samples required to be at a leaf node, the number of trees in the forest, and the splitting criterion.

Next, the training data is loaded from a CSV file and is split into input features and target variable. The code also includes an optional step for feature selection using SelectKBest, which is currently commented out. SelectKBest is a method for selecting the most relevant features from a dataset based on their relationship with the target variable. The RFC model is then instantiated, and the best set of hyperparameters is selected by running GridSearchCV with stratified shuffle split cross-validation.

The RFC model is then fitted to the training data using the best hyperparameters found by GridSearchCV. The model is trained to classify the target variable based on the input features. The classification algorithm creates decision trees, and these trees are used to classify new data points. The RFC model is then used to make predictions on a separate test dataset, and the output is written to a CSV file. Finally, the output file is generated, and the predicted class probabilities are stored in a dataframe.

In summary, this code demonstrates a basic approach to using the RFC algorithm for classification tasks. The code uses hyperparameter tuning and feature selection to optimize the performance of the model. The output file provides the predicted probabilities for the classes in the test dataset, which can be used to evaluate the model's performance.


---------------------------------------------------------------------------------------

Command  to run-

---------------------------------------------------------------------------------------



python 5_code.py [training file name] [test file name]



Eg:



If Input files are-> train.csv & test.csv



then command to run will be -> python 5_code.py train.csv test.csv
