# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 21:19:33 2019

@author: Dell
"""

import numpy as np
import matplotlib.pyplot as mpl
import pandas as pd
ds=pd.read_csv('mi.csv')

u=[]
matches=ds['MATCHES']
innings=ds['INNINGS']
cent=ds['100']
ft=ds['50']
avg=ds['Ave']


u=[np.sqrt(innings[i]/matches[i]) for i in range(0,8)]
v=[20*cent[i]+5*ft[i] for i in range(0,8)]
w=[.3*v[i]+.7*avg[i] for i in range(0,8)]
cs=[u[i]*w[i] for i in range(0,8)]
m=['33','35','20','20','18','3','27','29']
m=[int(m[i]) for i in range(8)]
ncs=[cs[i]/max(cs) for i in range(8)]
nm=[(m[i])/max(m) for i in range(8)]
bs=[.35*ncs[i]+.65*nm[i] for i in range(8)]
nbs=[bs[i]/max(bs) for i in range(8)]
#bowling
bow=ds['BOWLING']
wkt=ds['WKTS']
#wkt=[int(wkt[i]) for i in range(7)]
av=ds['Ave.1']
#av=[float(av[i]) for i in range(7)]
eco=ds['Econ']
fr=ds['4W']
#fr=[int(fr[i]) for i in range(7)]
fv=ds['5W']

u1=[np.sqrt(bow[i]/matches[i]) for i in range(0,8)]
v1=[10*fr[i]+wkt[i] for i in range(0,8)]
w1=[av[i]*eco[i] for i in range(0,8)]
for i in range(8):
  if w1[i]==float(0):
    w1[i]=1
#bbs=[(u1[i]*v1[i])/w1[i]  for i in range(0,7)]

bowsco=[u1[i]*v1[i]/w1[i] for i in range(8)]
nbowsco=[bowsco[i]/max(bowsco) for i in range(8)]

t_A=['RG SHARMA','DE COCK','HARDIK PANDYA','KIERON POLLARD']
t_B=['KRUNAL PANDYA','JASPRIT BUMRAH','SURYAKUMAR YADAV','ISHAN KKISHAN']
s=0
for i in range(4):
  s=s+nbs[i]
bat_s_a=s
s=0
for i in range(4,8):
  s=s+nbs[i]
bat_s_b=s
s=0
for i in range(4):
  s=s+nbowsco[i]
bow_s_a=s
s=0
s=0
for i in range(4,8):
  s=s+nbowsco[i]
bow_s_b=s
bow_s_b


strengthAbyB=(bat_s_a/bow_s_b)-(bat_s_b/bow_s_a)
dd=pd.read_csv('mumbai.csv')




