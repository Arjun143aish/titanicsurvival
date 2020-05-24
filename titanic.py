import os
import pandas as pd
import numpy as np

os.chdir("C:\\Users\\user\\Documents\\Python\\Kaggle Competition\\titanic")

Train = pd.read_csv("train.csv")
Test = pd.read_csv("test.csv")
submission = pd.read_csv("gender_submission.csv")

Train.isnull().sum()

M1 = Train.isnull().sum()

M1 = pd.DataFrame(M1)
M1.columns = ['NAs']
M1['Threshold @ 40%'] = Train.shape[0]*0.4
M1['Relevant Columns'] = np.where(M1['NAs'] >= M1['Threshold @ 40%'],'remove','retain')

Train.drop(['Cabin'], axis =1, inplace =True)

Train.drop(['Name','Ticket'], axis =1, inplace =True)

Train.isnull().sum()

Train.dtypes

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize = [12,8])
sns.boxplot(x = 'Pclass', y = 'Age',data= Train)

Train['Age'].fillna(Train.groupby('Pclass')['Age'].transform('mean'), inplace =True)
Train['Age'] = round(Train['Age'],2) 

mode = Train['Embarked'].mode()[0]
Train['Embarked'].fillna(mode, inplace =True)

Test.isnull().sum() 
Test.drop(['Name','Cabin','Ticket'], axis =1, inplace =True)

Test['Age'].fillna(Test.groupby('Pclass')['Age'].transform('mean'), inplace =True)
Test['Age'] = round(Test['Age'],2) 

fare = Test['Fare'].mean()
Test['Fare'].fillna(fare, inplace =True)

Category_Train = (Train.dtypes == 'object')
dummy_Train = pd.get_dummies(Train.loc[:,Category_Train],drop_first = True)
Train2 = pd.concat([Train.loc[:,~Category_Train],dummy_Train], axis =1)
 
Category_Test = (Test.dtypes == 'object')
dummy_Test = pd.get_dummies(Test.loc[:,Category_Test],drop_first = True)
Test2 = pd.concat([Test.loc[:,~Category_Test],dummy_Test], axis =1)

Train_X = Train2.drop(['Survived'], axis =1)
Train_Y = Train2['Survived'].copy()
Test_X = Test2.copy()
Test_Y = submission['Survived'].copy()

from statsmodels.api import Logit

M1_Model = Logit(Train_Y,Train_X).fit()
M1_Model.summary()

Test_pred = M1_Model.predict(Test_X)
Test['Test_pred'] = Test_pred
Test['Test_Class'] = np.where(Test['Test_pred'] > 0.5,1,0)

from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score

Con_Mat = confusion_matrix(Test['Test_Class'],Test_Y)
sum(np.diag(Con_Mat))/Test_Y.shape[0]*100

Submission = pd.DataFrame({'PassengerId': Test['PassengerId'], 'Survived': Test['Test_Class']})
filename = 'titanic_pred.csv'
Submission.to_csv(filename, index = 'False')

import pickle

pickle.dump(M1_Model,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
