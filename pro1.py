

import numpy as np
import matplotlib.pyplot as mpl
import pandas as pd
ds=pd.read_csv('rcbsrh.csv')


u=[]
matches=ds['MATCHES']
innings=ds['INNINGS']
cent=ds['100']
ft=ds['50']
avg=ds['Ave']
strike=ds['SR']


u=[np.sqrt(innings[i]/matches[i]) for i in range(0,22)]
v=[20*cent[i]+5*ft[i] for i in range(0,22)]
w=[.3*v[i]+.4*avg[i]+0.3*strike[i] for i in range(0,22)]
cs=[u[i]*w[i] for i in range(0,22)]
m=['31.3','36.6','21','11','14.3','1','17','31.3','5','3','13','53.2','28.3','11.3','14','2','19','16','7','0','0','1']
m=[float(m[i]) for i in range(22)]
ncs=[cs[i]/max(cs) for i in range(22)]
nm=[(m[i])/max(m) for i in range(22)]
bs=[.35*ncs[i]+.65*nm[i] for i in range(22)]
nbs=[bs[i]/max(bs) for i in range(0,22)]

#bowling
bow=ds['BOWLING']
wkt=ds['WKTS']
#wkt=[int(wkt[i]) for i in range(7)]
av=ds['Ave.1']
#av=[float(av[i]) for i in range(7)]
eco=ds['Econ']
fr=ds['4W']

for i in range(22):
  if bow[i]==float(0):
    bow[i]=1
#fr=[int(fr[i]) for i in range(7)]
fv=ds['5W']
player=ds['PLAYER']

u1=[np.sqrt(bow[i]/matches[i]) for i in range(0,22)]
v1=[float(wkt[i]/bow[i]) for i in range(0,22)]
w1=[av[i]*eco[i] for i in range(0,22)]
for i in range(22):
  if w1[i]==float(0):
    w1[i]=1
#bbs=[(u1[i]*v1[i])/w1[i]  for i in range(0,7)]

bowsco=[u1[i]*v1[i]/w1[i] for i in range(22)]
nbowsco=[bowsco[i]/max(bowsco) for i in range(22)]

t_A=[player[i] for i in range(0,11)]
t_B=[player[i] for i in range(11,22)]
s=0
for i in range(11):
  s=s+nbs[i]
bat_s_a=s
s=0
for i in range(11,22):
  s=s+nbs[i]
bat_s_b=s
s=0
for i in range(11):
  s=s+nbowsco[i]
bow_s_a=s
s=0
s=0
for i in range(11,22):
  s=s+nbowsco[i]
bow_s_b=s
bow_s_b


strengthAbyB=(bat_s_a/bow_s_b)-(bat_s_b/bow_s_a)





