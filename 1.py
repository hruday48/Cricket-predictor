# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 23:28:22 2019

@author: Dell
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 00:44:33 2019

@author: Dell
"""

import numpy as np

import pandas as pd
dataset=pd.read_csv('matcheslatest.csv')
dataset.drop(["result", "dl_applied","win_by_runs","win_by_wickets","id"], axis = 1, inplace = True) 
dataset=dataset.sort_values('season')

dataset.drop(dataset[dataset['team1'] =='Pune Warriors'].index, inplace = True) 
dataset.drop(dataset[dataset['team2'] =='Pune Warriors'].index, inplace = True) 
dataset.drop(dataset[dataset['season'] <=2012].index, inplace = True) 
dataset = dataset.reset_index(drop=True)

df1=dataset.pop('winner')
df2=dataset.pop('venue')
dataset['venue']=df2
dataset['winner']=df1



dataset[pd.isnull(dataset['winner'])]
dataset['winner'].fillna('Draw', inplace=True)
print(dataset.info())
print(dataset.isnull().values.sum())
print(dataset.isnull().sum())
dataset[pd.isnull(dataset['scoret2'])]
dataset['scoret2'].fillna(2, inplace=True)



dataset.replace(['Delhi Capitals'],['Delhi Daredevils'],inplace=True)
dataset.replace(['Rising Pune Supergiants'],['Rising Pune Supergiant'],inplace=True)
dataset.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Chennai Super Kings',
                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',
                 'Sunrisers Hyderabad','Rising Pune Supergiant']
                ,['MI','KKR','RCB','CSK','RR','DD','GL','KXIP','SRH','RPS'],inplace=True)
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
accu=['Naivebayes','SVC','Randomforest','knn','decisiontree']
labels = dataset['winner'].astype('category').cat.categories.tolist()
counts = dataset['winner'].value_counts()
sizes = [counts[var_cat] for var_cat in labels]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True) #autopct is show the % on plot
ax1.axis('equal')
plt.show()
encode = {'team1': {'MI':1,'KKR':2,'RCB':3,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11},
          'team2': {'MI':1,'KKR':2,'RCB':3,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11},
          'toss_winner': {'MI':1,'KKR':2,'RCB':3,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11},
          'winner': {'MI':1,'KKR':2,'RCB':3,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11, 'Draw':14}}
dataset.replace(encode, inplace=True)
dicVal = encode['winner']
print(dicVal['MI']) 
print(list(dicVal.keys())[list(dicVal.values()).index(1)]) #find key by value search 
df = pd.DataFrame(dataset)
df.apply(lambda x: sum(x.isnull()),axis=0) 
dataset.drop(["city"], axis = 1, inplace = True) 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset['toss_decision']=le.fit_transform(dataset['toss_decision'])
dataset['venue']=le.fit_transform(dataset['venue'])

df.dtypes 
dataset.dtypes

from sklearn.model_selection import train_test_split
x=dataset.iloc[:,[1,2,3,4,5,6,8]].values
y=dataset.iloc[:,9].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)



from sklearn.svm import SVC 
classifier=SVC(kernel='poly')
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



from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train, y_train)
y_prednb=classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
from sklearn import metrics
accuracy1 = metrics.accuracy_score(y_prednb,y_test)
print('Accuracy : %s' % '{0:.3%}'.format(accuracy1))










from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
acc1=[]

acc1.append(accuracy1)
acc1.append(accuracy2)
acc1.append(accuracy3)

acc1.append(accuracy5)
def classification_model(model, data, predictors, outcome):
  model.fit(data[predictors],data[outcome])
  
  predictions = model.predict(data[predictors])
  
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print('Accuracy : %s' % '{0:.3%}'.format(accuracy))

 


outcome_var = ['winner']
predictor_var = ['team1', 'team2', 'venue', 'toss_winner','toss_decision','scoret1','scoret2']





from sklearn.ensemble import RandomForestRegressor"""
outcome_var=['winner']
predictor_var = ['team1','team2','toss_winner','toss_decision','scoret1','scoret2']
model = LogisticRegression()
classification_model(model, dataset,predictor_var,outcome_var)""""
model = RandomForestClassifier(n_estimators=100)
outcome_var = ['winner']
predictor_var = ['team1', 'team2', 'venue', 'toss_winner','toss_decision','scoret1','scoret2']
classification_model(model, dataset,predictor_var,outcome_var)
"""
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
outcome_var = ['winner']
predictor_var = ['team1', 'team2', 'venue', 'toss_winner','toss_decision','scoret1','scoret2']
classification_model(model, dataset,predictor_var,outcome_var)
from sklearn.svm import SVC
model=SVC()
"""
outcome_var = ['winner']
predictor_var = ['team1', 'team2', 'venue', 'toss_winner','toss_decision','scoret1','scoret2']
classification_model(model, dataset,predictor_var,outcome_var)



from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=2)
outcome_var = ['winner']
predictor_var = ['team1', 'team2', 'venue', 'toss_winner','toss_decision','scoret1','scoret2']
classification_model(model, dataset,predictor_var,outcome_var)





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

accuracy1=32.7
accuracy2=41
accuracy3=59.24
accuracy7=71.29
accuracy5=50
acc1=[]
acc1.append(accuracy1)
acc1.append(accuracy2)
acc1.append(accuracy3)
acc1.append(accuracy7)
acc1.append(accuracy5)

plt.plot(accu,acc1)
plt.xlabel('accuracy')
plt.ylabel('value')
plt.show()





"""
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(criterion='entropy')
outcome_var = ['winner']
predictor_var = ['team1', 'team2', 'venue', 'toss_winner','toss_decision','scoret1','scoret2']
classification_model(model, dataset,predictor_var,outcome_var)"""


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(criterion='entropy')
model.fit(x_train, y_train)
y_prednb=model.predict(x_test)
from sklearn.metrics import confusion_matrix
from sklearn import metrics
accuracy5 = metrics.accuracy_score(y_prednb,y_test)
print('Accuracy : %s' % '{0:.3%}'.format(accuracy5))





"""from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=2)
classifier.fit(x_train, y_train)
y_prednb=classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
from sklearn import metrics
accuracy6 = metrics.accuracy_score(y_prednb,y_test)
print('Accuracy6 : %s' % '{0:.3%}'.format(accuracy6))"""

team1='RCB'
team2='KKR'
toss_winner='RCB'
input=[dicVal[team1],dicVal[team2],'6',dicVal[toss_winner],'1','9','7']
input = np.array(input).reshape((1, -1))
output=model.predict(input)
print(list(dicVal.keys())[list(dicVal.values()).index(output)]) 



team1='KXIP'
team2='RR'
toss_winner='RR'
input=[dicVal[team1],dicVal[team2],'24',dicVal[toss_winner],'1','9','9']
input = np.array(input).reshape((1, -1))
output=model.predict(input)
print(list(dicVal.keys())[list(dicVal.values()).index(output)])



team1='RCB'
team2='CSK'
toss_winner='CSK'
input=[dicVal[team1],dicVal[team2],'16',dicVal[toss_winner],'1','7','10']
input = np.array(input).reshape((1, -1))
output=model.predict(input)
print(list(dicVal.keys())[list(dicVal.values()).index(output)])



team1='KXIP'
team2='RR'
toss_winner='RR'
input=[dicVal[team1],dicVal[team2],'24',dicVal[toss_winner],'1','10','6']
input = np.array(input).reshape((1, -1))
output=model.predict(input)
print(list(dicVal.keys())[list(dicVal.values()).index(output)])


imp_input = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print(imp_input)











