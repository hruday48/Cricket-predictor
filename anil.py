# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 22:18:42 2019

@author: Dell
"""

import numpy as np

import pandas as pd
dataset=pd.read_csv('matches.csv')
dataset.drop(["result", "dl_applied","win_by_runs","win_by_wickets","id"], axis = 1, inplace = True) 
dataset=dataset.sort_values('season')
dataset = dataset.reset_index(drop=True)
df1=dataset.pop('winner')
df2=dataset.pop('venue')
dataset['venue']=df2
dataset['winner']=df1


print(dataset.info())
print(dataset.isnull().values.sum())
print(dataset.isnull().sum())
dataset.drop(['city'], axis = 1) 






dataset[pd.isnull(dataset['winner'])]
dataset['winner'].fillna('Draw', inplace=True)
dataset.replace(['Delhi Capitals'],['Delhi Daredevils'],inplace=True)
dataset.replace(['Rising Pune Supergiants'],['Rising Pune Supergiant'],inplace=True)


dataset.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',
                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',
                 'Sunrisers Hyderabad','Rising Pune Supergiant','Kochi Tuskers Kerala','Pune Warriors']
                ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW'],inplace=True)
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
winner_count = dataset['winner'].value_counts()
sns.set(style="darkgrid")
sns.barplot(winner_count.index, winner_count.values, alpha=0.9)
plt.title('Frequency Distribution of winnerss')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('winners', fontsize=12)
plt.show()
accu=['naivebayes','SVC','Randomforest']
labels = dataset['winner'].astype('category').cat.categories.tolist()
counts = dataset['winner'].value_counts()
sizes = [counts[var_cat] for var_cat in labels]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True) #autopct is show the % on plot
ax1.axis('equal')
plt.show()




encode = {'team1': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},
          'team2': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},
          'toss_winner': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},
          'winner': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13,'Draw':14}}
dataset.replace(encode, inplace=True)
dataset[pd.isnull(dataset['city'])]
dataset['city'].fillna('Dubai',inplace=True)
dicVal = encode['winner']
print(dicVal['MI']) 
print(list(dicVal.keys())[list(dicVal.values()).index(1)]) #find key by value search 
df = pd.DataFrame(dataset)
df.apply(lambda x: sum(x.isnull()),axis=0) 

var_mod = ['city','toss_decision','venue']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])
df.dtypes 
dataset.dtypes
from sklearn.model_selection import train_test_split
x=dataset.iloc[:,1:6].values
y=dataset.iloc[:,7].values


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train, y_train)
y_prednb=classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
from sklearn import metrics
accuracy1 = metrics.accuracy_score(y_prednb,y_test)
print('Accuracy : %s' % '{0:.3%}'.format(accuracy1))

from sklearn.svm import SVC 
classifier=SVC()
classifier.fit(x_train, y_train)
y_predsvc=classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
from sklearn import metrics
accuracy2 = metrics.accuracy_score(y_predsvc,y_test)
print('Accuracy : %s' % '{0:.3%}'.format(accuracy2))
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=790)
classifier.fit(x_train, y_train)
y_predrf=classifier.predict(x_test)
accuracy3 = metrics.accuracy_score(y_predrf,y_test)
print('Accuracy : %s' % '{0:.3%}'.format(accuracy3))
"""outcome_var = ['winner']
predictor_var = ['team1', 'team2', 'venue', 'toss_winner','city','toss_decision']
classification_model(model, df,predictor_var,outcome_var)"""

print('Accuracy : %s' % '{0:.3%}'.format(accuracy))
cm= confusion_matrix(y_test, y_pred)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
def classification_model(model, data, predictors, outcome):
  model.fit(data[predictors],data[outcome])
  
  predictions = model.predict(data[predictors])
  
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print('Accuracy : %s' % '{0:.3%}'.format(accuracy))

  kf = KFold(data.shape[0], n_folds=5)
  error = []
  for train, test in kf:
    train_predictors = (data[predictors].iloc[train,:])
    
    train_target = data[outcome].iloc[train]
    
    model.fit(train_predictors, train_target)
    
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
 
  print('Cross-Validation Score : %s' % '{0:.3%}'.format(np.mean(error)))

  model.fit(data[predictors],data[outcome]) 
  from sklearn.ensemble import RandomForestRegressor
outcome_var=['winner']
predictor_var = ['team1','team2','toss_winner']
model = LogisticRegression()
classification_model(model, dataset,predictor_var,outcome_var)
model = RandomForestClassifier(n_estimators=100)
outcome_var = ['winner']
predictor_var = ['team1', 'team2', 'venue', 'toss_winner','city','toss_decision']
classification_model(model, df,predictor_var,outcome_var)
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
outcome_var = ['winner']
predictor_var = ['team1', 'team2', 'venue', 'toss_winner','city','toss_decision']
classification_model(model, df,predictor_var,outcome_var)
from sklearn.svm import svc
model=svc()

outcome_var = ['winner']
predictor_var = ['team1', 'team2', 'venue', 'toss_winner','city','toss_decision']
classification_model(model, df,predictor_var,outcome_var)




from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
winner_count = dataset['winner'].value_counts()
sns.set(style="darkgrid")
sns.barplot(winner_count.index, winner_count.values, alpha=0.9)
plt.title('Frequency Distribution of winnerss')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('winners', fontsize=12)
plt.show()

acc1=[]
acc1.append(accuracy1)
acc1.append(accuracy2)
acc1.append(accuracy3)
plt.plot(accu,acc1)
plt.xlabel('accuracy')
plt.ylabel('value')
plt.show()



