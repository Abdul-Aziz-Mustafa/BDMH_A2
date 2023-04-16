

# step1 imorting all the nesssary libraries
import argparse
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, precision_recall_curve
# from sklearn.metrics import auc, plot_precision_recall_curve
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from sklearn.ensemble import RandomForestClassifier
# from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import AdaBoostClassifier
# from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
# import xgboost as xgb
from sklearn.model_selection import cross_validate
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
# from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.ensemble import GradientBoostingRegressor


# step2 reading the the training data from user provided arguemnt from the terminal

#definng parser for getting argument at cli
parser = argparse.ArgumentParser()
#adding argument which is path of train data
parser.add_argument('-tr', '--train', type=str, help='It is the path to a train file')
#adding argument which is path of test data
parser.add_argument('-te', '--test', type=str, help='It is the path to a test file')
#parsing the arg
args = parser.parse_args()

if not args.train:
    sys.exit(" Please specify the correct path to the train file ")

try:
    training_data = pd.read_csv(args.train)
except FileNotFoundError:
    sys.exit("Could not find the file")
# training_data=pd.read_csv('kaggle_train.csv')


y=training_data['Labels']
training_data=training_data.drop(['Labels'],axis=1)


# step3- defined function for evaulating the best model after all trials

def evaluate_model(model, X, y):
  cv =RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
  # y_pred = cross_val_predict(model, X, y, cv=10)
  scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
  # return y_pred
  return scores

# step4-defining the stacking classifier (an ensemble method in machine learning)

def get_stacking():
 # define the base models
 level0 = list()
 level0.append(('rf', RandomForestClassifier()))
 level0.append(('et', ExtraTreesClassifier()))
#  level0.append(('gb',GradientBoostingClassifier()))
#  level0.append(('lr', LogisticRegression()))
#  level0.append(('knn', KNeighborsClassifier()))
#  level0.append(('abc',AdaBoostClassifier(n_estimators=400, random_state=0)))
#  level0.append(('svm', SVC()))
#  level0.append(('bayes', GaussianNB()))
 # define meta learner model
 level1 = LogisticRegression()
 # define the stacking ensemble
 model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
 return model


#step5-defining model kendal function for feature selection
def Model_kendall(X1, y):
  X = X1
  cor = X.corr(method='kendall') # Calculate Correlation between each variable
  cor_target = abs(y) #Correlation with output variable
  relevant_features = cor_target[cor_target>=0.1]
  # kendall=X.columns[relevant_features.get_support(indices=True)] #Selecting correlated features above than the given threshold
  kendall_sc= pd.DataFrame(relevant_features).index #This saves the columns names in sc variable
  #This will show the first
  kendall=X.iloc[:,kendall_sc]
  return [kendall,kendall_sc]


# step6-defining the model_Etc function for best features usi
def Model_ETC(X1, y):
    X=X1
    clf =AdaBoostClassifier()
    clf = clf.fit(X, y)
    model = SelectFromModel(clf, prefit=True)
    sf = model.transform(X)
    ET_sc = X.columns[model.get_support(indices=True)]
    etC=X[ET_sc]
    return [etC,ET_sc]


from sklearn.feature_selection import RFE

from sklearn.feature_selection import SelectFromModel

# step7 finding the best features using RGE algorithm 
def Model_LinearSVC(X1, y):
    X=X1
    lr = AdaBoostClassifier()
    rfe = RFE(lr) #It assigns weights to features using logistic regression
    sf = rfe.fit(X, y)  # It learns relationship and transfrom the data

    l1_sc = X.columns[rfe.get_support(indices=True)]
    l1C=X[l1_sc]
    return [l1C,l1_sc]  

# step8 this Model_VarThreshold function remove all zrro variance from thre input data

from sklearn.feature_selection import VarianceThreshold #Import the library
def Model_VarThreshold(X1):
    sel = VarianceThreshold(1) # This will remove all zero-variance features
    sf = sel.fit_transform(X1) # The file will be saved in sf
    var_sc = X1.columns[sel.get_support(indices=True)] #This saves the columns names in sc variable
    var_th=X1[var_sc]
    return var_th
etC=Model_ETC(training_data,y)
# etC[0]
# etC= PCA(n_components=10).fit_transform(training_data) 
# etC
# X_train, X_test, y_train, y_test = train_test_split(etC[0], y, test_size=0.11, random_state=19)


# step 9 we also checked which model is giving the best accuracy for input model by runnning 
# LazyClassifier on our iinput code


# clf = LazyClassifier(verbose=0,
#                      ignore_warnings=True, 
#                      custom_metric=None)
# models, predictions = clf.fit(X_train, X_test, y_train, y_test)
# models


from sklearn.svm import SVC
# initialize the base classifier
# base_cls = RandomForestClassifier()
# seed = 8
# # no. of base classifier
# num_trees = 500
 
# bagging classifier
# step 10 finally founded the best classfier using evaluate model function and running it on the testing dataset
model1 =GradientBoostingClassifier(n_estimators=3000)
 
mp={'abc':GradientBoostingClassifier()}
# 'NC': GradientBoostingClassifier(n_estimators=580),'RFNew':GradientBoostingClassifier(n_estimators=680),'LRNew':LogisticRegression(),'KNNew':GradientBoostingClassifier(n_estimators=880),'GNBNew':GradientBoostingClassifier(min_samples_leaf=1,min_samples_split=3,criterion='squared_error',loss='deviance',n_estimators=130,random_state=10,max_features='auto'),'ETNew':GradientBoostingClassifier(n_estimators=1180),'SVCNew':GradientBoostingClassifier(n_estimators=1880)}



for key, model in mp.items():
  ans=evaluate_model(model,etC[0], y)
  print(key)
  print(ans.mean())

model=GradientBoostingClassifier(n_estimators=880)

training_data


# model=GradientBoostingClassifier()
model.fit(etC[0],y)



# step11 generating the the output files from the most optimsed model
if not args.test:
    sys.exit("Please specify the correct path to the train file ")

try:
    test_data = pd.read_csv(args.test)
except FileNotFoundError:
    sys.exit("Could not find the file")

# test_data=pd.read_csv('kaggle_test.csv')
testdata=test_data[etC[1]]
tX=testdata
# tX = pd.DataFrame(scaler.fit_transform(testdata))
# tX.columns=testdata.columns
# tX
id=test_data["ID"]
predictionsTest=model.predict(tX)

Y_prob = [x[1] for x in model.predict_proba(tX)]
df = pd.DataFrame(id, columns = ["ID"])
df2= pd.DataFrame(id, columns = ["ID"])
df["Labels"]=Y_prob
df2["Labels"]=predictionsTest
df2.to_csv('output.csv',index=False)

df.to_csv('5_output.csv',index=False)
# final main prediction results are generated in output.csv

