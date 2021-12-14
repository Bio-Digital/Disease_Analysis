
# From Kaggle db: https://www.kaggle.com/debasisdotcom/parkinson-disease-detection 

from itertools import count
from matplotlib.pyplot import cla, sca
from numpy import NaN, array, dtype
import numpy as np
from numpy.typing import _128Bit
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model._glm.glm import _y_pred_deviance_derivative
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import SGDRegressor, PassiveAggressiveClassifier, LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.metrics import accuracy_score


import seaborn as sb

import decimal

df = pd.read_csv(r"C:\Users\Freemason\Documents\code\TestAix\PD\Analysing_pd_Voice\PD_voice.csv")


df_corr = df.corr()

# Find highly correlated features and drop them
highly_correlated_features = set()

for feature_column in range(0, len(df_corr)):
   if feature_column == 'status':
     continue
   feature_column_name = df_corr.columns[feature_column]
   for feature_row in range(0,len(df_corr.index)):
       feature_row_name = df_corr.index[feature_row]
       if feature_row_name == feature_column_name:
           continue
       corr_value = df_corr.iloc[feature_column][feature_row]
       if corr_value > 0.67:
           highly_correlated_features.add(feature_row_name)
print(highly_correlated_features)
df = df.drop(highly_correlated_features, axis= 1)







# vocalFreq_X = df.loc[: ,('MDVP:Fo(Hz)','MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)')] #Only Selecting the columns that contain the vocal frequency
X = df.drop(['status', 'name'], axis=1).values 
# vocalFreq_X = vocalFreq_X.to_numpy()
Y = df['status'].values

# boxp = df.boxplot(column=['MDVP:Fo(Hz)','MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)'])


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, shuffle= True, random_state=0)

# print(vocalFreq_X.head())

# VocalFreq_model = RandomForestClassifier()
# VocalFreq_model.fit(VocalFreq_X_train, VocalFreq_Y_train)






# for est in range(5, 60, 5):
#     for depth in range(5,20, 2):
#         classifier = RandomForestClassifier(n_estimators=est, max_depth=depth)
#         classifier.fit(X_train, Y_train)
#         y_train_pred = classifier.predict(X_train)
#         y_test_pred = classifier.predict(X_test)

#         accuracy_train = accuracy_score(Y_train, y_train_pred)
#         accuracy_test = accuracy_score(Y_test, y_test_pred)
#         if accuracy_test > 0.98 and accuracy_train - accuracy_test < 0.5:
#             print('est: ' + str(est) + ', depth: ' + str(depth))
#             print('Accuracy\t\ttrain: %.4f , test: %.4f' %(accuracy_train, accuracy_test))





n_fold = 5

fold = StratifiedKFold(n_splits = n_fold, random_state = 0, shuffle = True)

accuracy = 0

for est in range(5, 60, 5): # starts at 5, increments by 5 till 60
    for depth in range(5,20,2):

        scores = []

        #Iterate over each step
        for train_index, test_index in fold.split(X, Y):

            #Calculate X/Y train of current iteration
            X_train, Y_train = X[train_index], Y[train_index]

            #Calculate X/Y train of current iteration
            X_test, Y_test = X[test_index], Y[test_index]

            #Create a new model 
            classifier = RandomForestClassifier(n_estimators=est, max_depth=depth)
            classifier.fit(X_train, Y_train)
            #Predict out of Fold data(Test)
            pred = classifier.predict(X_test)

            #Calculate score on Out of Fold data (test)
            score = accuracy_score(Y_test , pred)
            scores.append(score)
        
        scores = np.array(scores)
        print('est: ' + str(est) + ', depth: ' + str(depth))
        print('Accuracy average on test set: %.4f std.dev.: %.4f' %(scores.mean(), scores.std()))


# scaler = MinMaxScaler()

# X_train_Scaled = scaler.fit_transform(VocalFreq_X_train)
# X_test_Scaled = scaler.transform(VocalFreq_X_test)

# X_train_Scaled.shape

# ypred = model.predict(X_test)
# predVal = [88.333, 112.24, 84.072] # pd = 1 has parkinsons


# predVal = [241.404, 248.834, 232.483] # pd = 0 healthy
# VocalFreq_ypred = VocalFreq_model.predict([predVal])
# print(VocalFreq_ypred)







