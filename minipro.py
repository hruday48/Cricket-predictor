# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 21:13:51 2019

@author: Dell
"""

import numpy as np

import pandas as pd
dataset=pd.read_csv('matcheslatest.csv')
dataset.drop(["result", "dl_applied","win_by_runs","win_by_wickets","id"], axis = 1, inplace = True) 
dataset=dataset.sort_values('season')
dataset = dataset.reset_index(drop=True)
dataset.drop(dataset[dataset['team1'] =='Kochi Tuskers Kerala'].index, inplace = True) 
dataset.drop(dataset[dataset['team1'] =='Pune Warriors'].index, inplace = True) 
dataset.drop(dataset[dataset['season'] <=2012].index, inplace = True) 



df1=dataset.pop('winner')
df2=dataset.pop('venue')

dataset['venue']=df2
dataset['winner']=df1
d1=dataset.pop('Unnamed:15')

print(dataset.info())
print(dataset.isnull().values.sum())
print(dataset.isnull().sum())
dataset[pd.isnull(dataset['winner'])]
dataset['winner'].fillna('Draw', inplace=True)
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

accu=['naivebayes','SVC','Randomforest']
labels = dataset['winner'].astype('category').cat.categories.tolist()
counts = dataset['winner'].value_counts()
sizes = [counts[var_cat] for var_cat in labels]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True) #autopct is show the % on plot
ax1.axis('equal')
plt.show()
encode = {'team1': {'MI':1,'KKR':2,'RCB':3,'CSK':4,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11},
          'team2': {'MI':1,'KKR':2,'RCB':3,'CSK':4,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,},
          'toss_winner': {'MI':1,'KKR':2,'RCB':3,'CSK':4,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11},
          'winner': {'MI':1,'KKR':2,'RCB':3,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'Draw':14}}
dataset.replace(encode, inplace=True)


















