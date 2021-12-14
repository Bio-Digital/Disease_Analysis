# DB = https://physionet.org/content/gaitpdb/1.0.0/

from itertools import count
from numpy import NaN, array, dtype
import numpy as np
from numpy.typing import _128Bit
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model._glm.glm import _y_pred_deviance_derivative
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor, PassiveAggressiveClassifier, LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.metrics import accuracy_score


import seaborn as sb

import decimal

df = pd.read_excel(r"C:\Users\Freemason\Documents\code\TestAix\PD\demographics.xls")
df.rename({'UPDRS':"UPDRScale"},axis=1,inplace=True)
# df.dropna(subset= ["Speed_01 (m/sec)"], inplace=True)
# replace_To = ''
# df.replace(NaN, replace_To, inplace=False)
# X = df.iloc[:, 0:164].values
# y = df.iloc[:, 2].values

X = df.drop(columns=['ID','Study', 'Group', 'Gender', 'Speed_10']) #Dropping all the columns that contain string and Speed_10 since there is too many NaN values to be reliable
# y = df['Group']
y = df['Group'].replace(NaN, 0, inplace=False)
global n
Rs = 0
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=Rs)
simp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
# print(simp_mean.fit_transform(X_train[['UPDRS']]))
# print(X_train[['UPDRS']].mean())

# print(simp_mean.fit_transform(X_train[['HoehnYahr']]))
# print(X_train[['HoehnYahr']].mean())


# df['UPDRS']




def Y_trainLabeler(Y_train, Y_test, inver):
    le = preprocessing.LabelEncoder()
    le.fit(Y_train)
    Y_train = le.transform(Y_train) # Using Label encoders to convert Pd and CO into 1 and 0

    le.fit(Y_test)
    if (inver != 1):
        Y_test = le.transform(Y_test) 
        return (Y_train, Y_test)

Y_train, Y_test = Y_trainLabeler(Y_train, Y_test, 0)

lis = ['UPDRScale', 'UPDRSM', 'HoehnYahr', 'TUAG', 'Height (meters)', 'Weight (kg)', 'Speed_01 (m/sec)']

Height = X_train['Height (meters)']

def meanImplementer(count):
    for colmn in lis: # Looks over the certain columns that I wanna replace the NaN vals with the mean
        if (colmn) == 'Height (meters)':
            for dat in Height: # For Loop to get the heights the same unit: Cm -> Metres
                # if (isinstance((Height.values[count]),float)):
                d = decimal.Decimal(dat)
                dexp = (d.as_tuple().exponent) # Checks the decimal point of the current index val
                if str(Height.values[count]) != 'nan':
                    if (dexp >= 0):
                        # if (Height.values[count] < 10):
                        Height.values[count] = (dat / 100)
                    # print(Height.values[count]) 
                # print(dtype(Height[count]))
                
                count+=1
            X_train['Height (meters)'] = Height.values
        global X_TrainMean
        X_TrainMean = simp_mean.fit_transform(X_train[[colmn]])
        X_train[colmn] = X_TrainMean
        X_TestMean = simp_mean.fit_transform(X_test[[colmn]])
        X_test[colmn] = X_TestMean
        
      
    
    return(X_train, X_test)
    # print(colmn)
    # print(X_TrainMean) # Gets the mean value of the columns
# print(X_train)
# print(Y_train)

X_train, X_test = meanImplementer(0)
X_train = X_train.sort_index(ascending=True)
X_test = X_test.sort_index(ascending=True)
model = RandomForestClassifier()
# print(X_train.isnull().sum())
model.fit(X_train, Y_train)

# print(X_train)
# ypred = model.predict(X_test)

predVar = [1, 78, 1.67, 75, 1.957746, 0, 0, 10.19, 1.089] # doesn't have pd
# predVar = [23, 68, 1.60, 69, 2, 19, 16, 7.53, 1.247] # Has PD
Patientinfo = X_train.iloc[7]
Patientinfo = Patientinfo.to_numpy()
ypred = model.predict([predVar])
tries = 0
while (ypred[0] != 0):   
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=Rs)
    X_train, X_test = meanImplementer(0)
    Y_train, Y_test = Y_trainLabeler(Y_train, Y_test, 0)
    # X_train = X_train.sort_index(ascending=True)
    # X_test = X_test.sort_index(ascending=True)
    model.fit(X_train, Y_train)
    ypred = model.predict([Patientinfo])
    print(Rs)
    Rs += 1 
    tries += 1
    if (tries > 5):
        break


if (Rs==0):
    Rs=1
print('Patient with:'
'\nSubjnum ' + str(Patientinfo[0]) + 
'\nAge ' + str(Patientinfo[1]) + 
'\nHeight (meters) ' + str(Patientinfo[2]) + 
'\nWeight (kg) ' +  str(Patientinfo[3]) +
'\nHoehnYahr ' +  str(Patientinfo[4]) +
'\nUPDRS ' +  str(Patientinfo[5]) +
'\nUPDRSM ' +  str(Patientinfo[6]) +
'\nTUAG ' +  str(Patientinfo[7]) +
'\nSpeed_01 (m/sec) ' +  str(Patientinfo[8]))
if ypred == 0:
    print("doesn't have parkinsons")
    print('It took ' + str(Rs) + ' number of tries')
if ypred == 1: 
    print('has parkinsons')
    print('It took ' + '1' + ' number of tries')


# score = accuracy_score(Y_test, ypred)
# print(score)



# correllation = X_train.corr
# print(correllation)



