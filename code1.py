from tkinter import * 
from PIL import ImageTk,Image 
import os
import sys
import numpy 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn import svm
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


window = Tk()
 
window.title("Welcome to the CAD prediction ")
 
window.geometry('700x450')
lbl = Label(window, text="are you excited for knowing the accuracy ->")
nos=StringVar()

#t1=Entry(window,textvariable=nos).grid(column=1,row=1,pady=10)
lbl.grid(column=0, row=0,padx=0,pady=10)


def clicked():
    

    df = pd.read_csv("data.csv")
    df
    y = df['Cath']
    
    df =df.drop(["Cath"],axis =1 )
    df.head()
    data=df
    data1 = y
    
    d1=data.values
    d2=data1.values
    
    
    ohe= OneHotEncoder(categorical_features=[4])
    data = df
    values = array(data)
    print(values)
    # integer encode
    
    
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(d1[:,3])
    onehot_encoder1 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder1.fit_transform(integer_encoded)
    ds=pd.DataFrame({'Male':onehot_encoded[:,1],'Female':onehot_encoded[:,0]})
    
    
    
    label_encoder2 = LabelEncoder()
    integer_encoded = label_encoder2.fit_transform(d1[:,10])
    onehot_encoder2 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder2.fit_transform(integer_encoded)
    ds1=pd.DataFrame({'Obe N':onehot_encoded[:,0],'Obe Y':onehot_encoded[:,1]})
    
    
    label_encoder3 = LabelEncoder()
    integer_encoded = label_encoder3.fit_transform(d1[:,11])
    onehot_encoder3 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder3.fit_transform(integer_encoded)
    ds2=pd.DataFrame({'CRF N':onehot_encoded[:,0],'CRF Y':onehot_encoded[:,1]})
    
    
    
    label_encoder4 = LabelEncoder()
    integer_encoded = label_encoder4.fit_transform(d1[:,12])
    onehot_encoder4 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder4.fit_transform(integer_encoded)
    ds3=pd.DataFrame({'CVA N':onehot_encoded[:,0],'CVA Y':onehot_encoded[:,1]})
    
    label_encoder5 = LabelEncoder()
    integer_encoded = label_encoder5.fit_transform(d1[:,13])
    onehot_encoder5 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder5.fit_transform(integer_encoded)
    ds4=pd.DataFrame({'AID N':onehot_encoded[:,0],'AID Y':onehot_encoded[:,1]})
    
    
    # In[98]:
    
    
    label_encoder6= LabelEncoder()
    integer_encoded = label_encoder6.fit_transform(d1[:,14])
    onehot_encoder6= OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder6.fit_transform(integer_encoded)
    ds5=pd.DataFrame({'TD N':onehot_encoded[:,0],'TD Y':onehot_encoded[:,1]})
    
    
    # In[99]:
    
    
    label_encoder7 = LabelEncoder()
    integer_encoded = label_encoder7.fit_transform(d1[:,15])
    onehot_encoder7 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder7.fit_transform(integer_encoded)
    ds6=pd.DataFrame({'CHF N':onehot_encoded[:,0],'CHF Y':onehot_encoded[:,1]})
    
    
    # In[100]:
    
    
    label_encoder8 = LabelEncoder()
    integer_encoded = label_encoder8.fit_transform(d1[:,16])
    onehot_encoder8 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder8.fit_transform(integer_encoded)
    ds7=pd.DataFrame({'DLP N':onehot_encoded[:,0],'DLP Y':onehot_encoded[:,1]})
    
    
    # In[101]:
    
    
    label_encoder9 = LabelEncoder()
    integer_encoded = label_encoder9.fit_transform(d1[:,20])
    onehot_encoder9 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder9.fit_transform(integer_encoded)
    ds8=pd.DataFrame({'WPP N':onehot_encoded[:,0],'WPP Y':onehot_encoded[:,1]})
    
    
    # In[102]:
    
    
    label_encoder10 = LabelEncoder()
    integer_encoded = label_encoder10.fit_transform(d1[:,21])
    onehot_encoder10 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder10.fit_transform(integer_encoded)
    ds9=pd.DataFrame({'Lung rales N':onehot_encoded[:,0],'Lung rales Y':onehot_encoded[:,1]})
    
    
    # In[103]:
    
    
    label_encoder11 = LabelEncoder()
    integer_encoded = label_encoder11.fit_transform(d1[:,22])
    onehot_encoder11 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder11.fit_transform(integer_encoded)
    ds10=pd.DataFrame({'SM N':onehot_encoded[:,0],'SM Y':onehot_encoded[:,1]})
    
    
    # In[104]:
    
    
    label_encoder12 = LabelEncoder()
    integer_encoded = label_encoder12.fit_transform(d1[:,23])
    onehot_encoder12 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder12.fit_transform(integer_encoded)
    ds11=pd.DataFrame({'DM N':onehot_encoded[:,0],'DM Y':onehot_encoded[:,1]})
    
    
    # In[105]:
    
    
    label_encoder13 = LabelEncoder()
    integer_encoded = label_encoder13.fit_transform(d1[:,25])
    onehot_encoder13 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder13.fit_transform(integer_encoded)
    ds12=pd.DataFrame({'Dyspnea N':onehot_encoded[:,0],'Dyspnea Y':onehot_encoded[:,1]})
    
    
    # In[106]:
    
    
    label_encoder14 = LabelEncoder()
    integer_encoded = label_encoder14.fit_transform(d1[:,27])
    onehot_encoder14 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder14.fit_transform(integer_encoded)
    ds13=pd.DataFrame({'Atypical N':onehot_encoded[:,0],'Atypical Y':onehot_encoded[:,1]})
    
    
    # In[107]:
    
    
    label_encoder15 = LabelEncoder()
    integer_encoded = label_encoder15.fit_transform(d1[:,28])
    onehot_encoder15 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder15.fit_transform(integer_encoded)
    ds14=pd.DataFrame({'Nonanginal N':onehot_encoded[:,0],'Nonanginal Y':onehot_encoded[:,1]})
    
    
    # In[108]:
    
    
    label_encoder16 = LabelEncoder()
    integer_encoded = label_encoder16.fit_transform(d1[:,30])
    onehot_encoder16= OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder16.fit_transform(integer_encoded)
    ds15=pd.DataFrame({'Exertional CP N':onehot_encoded[:,0],'Exertional CP Y':onehot_encoded[:,1]})
    
    
    # In[109]:
    
    
    label_encoder17 = LabelEncoder()
    integer_encoded = label_encoder17.fit_transform(d1[:,30])
    onehot_encoder17= OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder17.fit_transform(integer_encoded)
    ds16=pd.DataFrame({'LOwTh ang N':onehot_encoded[:,0],'LowTh angY':onehot_encoded[:,1]})
    
    
    # In[110]:
    
    
    label_encoder18 = LabelEncoder()
    integer_encoded = label_encoder18.fit_transform(d1[:,35])
    onehot_encoder18 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder18.fit_transform(integer_encoded)
    ds17=pd.DataFrame({'LVH N':onehot_encoded[:,0],'LVH Y':onehot_encoded[:,1]})
    
    
    # In[111]:
    
    
    label_encoder19 = LabelEncoder()
    integer_encoded = label_encoder19.fit_transform(d1[:,36])
    onehot_encoder19 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder19.fit_transform(integer_encoded)
    ds18=pd.DataFrame({'Poor R progression N':onehot_encoded[:,0],'Poor R Progression Y':onehot_encoded[:,1]})
    
    
    # In[112]:
    
    
    label_encoder20 = LabelEncoder()
    integer_encoded = label_encoder20.fit_transform(d1[:,37])
    onehot_encoder20 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder20.fit_transform(integer_encoded)
    ds19=pd.DataFrame({'LBBB':onehot_encoded[:,0],' BBB N':onehot_encoded[:,1],'RBBB':onehot_encoded[:,2]})
        
    
    
    # In[113]:
    
    
    label_encoder21 = LabelEncoder()
    integer_encoded = label_encoder21.fit_transform(d1[:,54])
    onehot_encoder21 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder21.fit_transform(integer_encoded)
    ds20=pd.DataFrame({'Moderate':onehot_encoded[:,0],'VHD N':onehot_encoded[:,1],'mild':onehot_encoded[:,3],'Severe':onehot_encoded[:,2]})
    
    
    # In[114]:
    
    
    lencod = LabelEncoder()
    label = lencod.fit_transform(d2[:,])
    #ds21=pd.DataFrame({'Cad':onehot_encoded[:,0],'Normal':onehot_encoded[:,1]})
    y = pd.DataFrame({'Cath':label[:,]})
    
    
    # In[115]:
    
    
    df1 = pd.concat([df,ds,ds1,ds2,ds3,ds4,ds5,ds6,ds7,ds8,ds9,ds10,ds11,ds12,ds13,ds14,ds15,ds16,ds17,ds18,ds19,ds20,y], axis =1)
    df1.head()
    
    
    # In[116]:
    
    
    df1.drop(["Sex","Obesity","CRF","CVA","Airway disease","Thyroid Disease","CHF","DLP","Weak Peripheral Pulse","Lung rales","Systolic Murmur","Diastolic Murmur","Dyspnea","Atypical","Nonanginal","Exertional CP","LowTH Ang","LVH","Poor R Progression","BBB","VHD"],axis=1, inplace = True)
    
    
    # In[117]:
    
    
    df3 =df1.copy()
    
    df2 = df1[["Age","Weight","Length","BMI","BP","PR","Function Class","FBS","TG","LDL","HDL","BUN","ESR","HB","K","Na","WBC","Lymph","Neut","PLT","EF-TTE","Region RWMA"]]
    df1 = df1.drop(["Age","Weight","Length","BMI","BP","PR","Function Class","FBS","TG","LDL","HDL","BUN","ESR","HB","K","Na","WBC","Lymph","Neut","PLT","EF-TTE","Region RWMA"], axis =1)
    df1.head()
    
    
    # In[118]:
    
    
    # In[119]:
    
    
    dfdata = pd.read_csv('final.csv')
    dfdata
    y=dfdata['Cath']
    dfdata=dfdata.drop(['Cath'],axis=1)
    y.head()
    dfdata.shape
    len(y)
    
    
    # In[120]:
    X_trainAndTest, X_validation, y_trainAndTest, y_validation = train_test_split(dfdata,y, test_size=0.20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_trainAndTest, y_trainAndTest, test_size=0.20, random_state=42)
        
    def crossover(col):
        a,b=random.randrange(0,80),random.randrange(0,80)
        for i in range(0,99):
            temp=i*2
            arr=[]
            arr=col[temp]
            for j in range(a,b):
                col[temp][j]=col[temp+1][j]
            for j in range(a,b):
                col[temp+1][j]=col[j]
        
    
    
    # In[121]:
    
    
    def mutation(col):
        for i in range(0,len(col)):   
            for j in range(0,16):
                a=random.randrange(0,80)
                b=random.randrange(0,100)
                if b<21:
                    if col[i][j]==1:
                        col[i][j]=0
                    if col[i][j]==0:
                        col[i][j]=1
    
    
    # In[122]:
    
    
    def selection(col):
        temp=[]
        for i in range(0,200):
            a,b=random.randrange(0,200),random.randrange(0,200)
            acc1,acc2=lst[a],lst[b]
            c=random.randrange(0,100)
            if c<21:
                if acc1<acc2:
                    temp.append(col[a])
                if acc2<acc1:
                    temp.append(col[b])
            if c>=21:
                if acc1>acc2:
                    temp.append(col[a])
                if acc2>acc1:
                    temp.append(col[b])
        for i in range(0,len(temp)):
            col[i]=temp[i]
    
    
    # In[123]:
    
    
    def calc_fitness(mat):
        for i in range(0,len(mat)):
            arr=[]
            arr.append(mat[i])
            df1 =dfdata
            #df2= X_test
            for j in range(0,len(arr)):
                if arr[j]==0:
                    df1.drop([df1.columns[j]],axis=1,inplace=True)
                    #df2.drop([df2.columns[j]],axis=1,inplace=True)
            x_tr, x_te, y_tr, y_te = train_test_split(df1,y, test_size=0.20)
            clf = svm.SVC(kernel='linear', C = 1.0, degree=3)
            clf.fit(x_tr,y_tr)
            predict=clf.predict(x_te)
            temp=accuracy_score(y_te,predict)
            temp1=f1_score(y_te,predict)
            temp=temp*100
            temp1=temp1*100
            lst.append(temp)
            lst1.append(temp1)
        
        
        
    
    
    # In[124]:
    
    
    col=[]
    
    for k in range (0,200):
        row=[]
        for i in range(0,80):
            j=random.choice([0,1])
            row.append(j)
        col.append(row)
    lst=[]
    lst1=[]
    calc_fitness(col)
    lst
    
    
    # In[125]:
    
    
    gen_acc_lst=[]
    gen_lst=[]
    gen_acc_lst1=[]
    gen_lst1=[]
    for i in range(0,20):
        selection(col)
        crossover(col)
        mutation(col)
        lst=[]
        lst1=[]
        calc_fitness(col)
        maxpos = lst.index(max(lst))
        gen_lst.append(col[maxpos])
        gen_acc_lst.append(max(lst))
        maxpos1 = lst1.index(max(lst1))
        gen_lst1.append(col[maxpos1])
        gen_acc_lst1.append(max(lst1))
        
    
    # In[126]:
    
    
    print(gen_acc_lst)
    print(gen_acc_lst1)
    ma=max(gen_acc_lst)
    mb=min(gen_acc_lst)
    mc=max(gen_acc_lst1)
    md=min(gen_acc_lst1)
    m1 = (ma+mb)/2
    m2 = (mc+md)/2
    plt.plot(gen_acc_lst)
    plt.plot(gen_acc_lst1)
    plt.savefig("ge1.png")
    lbl.configure(text="the SVC accuracy and f1_score is: "+str(m1) + "," + str(m2))

    def grph():
    #import pp.py as pp
    #pp.pras()
        import pp
        exec('pp.py')
       # os.system("pp.py")
def clicked12():
    

    df = pd.read_csv("data.csv")
    df
    y = df['Cath']
    
    df =df.drop(["Cath"],axis =1 )
    df.head()
    data=df
    data1 = y
    
    d1=data.values
    d2=data1.values
    
    
    ohe= OneHotEncoder(categorical_features=[4])
    data = df
    values = array(data)
    print(values)
    # integer encode
    
    
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(d1[:,3])
    onehot_encoder1 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder1.fit_transform(integer_encoded)
    ds=pd.DataFrame({'Male':onehot_encoded[:,1],'Female':onehot_encoded[:,0]})
    
    
    
    label_encoder2 = LabelEncoder()
    integer_encoded = label_encoder2.fit_transform(d1[:,10])
    onehot_encoder2 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder2.fit_transform(integer_encoded)
    ds1=pd.DataFrame({'Obe N':onehot_encoded[:,0],'Obe Y':onehot_encoded[:,1]})
    
    
    label_encoder3 = LabelEncoder()
    integer_encoded = label_encoder3.fit_transform(d1[:,11])
    onehot_encoder3 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder3.fit_transform(integer_encoded)
    ds2=pd.DataFrame({'CRF N':onehot_encoded[:,0],'CRF Y':onehot_encoded[:,1]})
    
    
    
    label_encoder4 = LabelEncoder()
    integer_encoded = label_encoder4.fit_transform(d1[:,12])
    onehot_encoder4 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder4.fit_transform(integer_encoded)
    ds3=pd.DataFrame({'CVA N':onehot_encoded[:,0],'CVA Y':onehot_encoded[:,1]})
    
    label_encoder5 = LabelEncoder()
    integer_encoded = label_encoder5.fit_transform(d1[:,13])
    onehot_encoder5 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder5.fit_transform(integer_encoded)
    ds4=pd.DataFrame({'AID N':onehot_encoded[:,0],'AID Y':onehot_encoded[:,1]})
    
    
    # In[98]:
    
    
    label_encoder6= LabelEncoder()
    integer_encoded = label_encoder6.fit_transform(d1[:,14])
    onehot_encoder6= OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder6.fit_transform(integer_encoded)
    ds5=pd.DataFrame({'TD N':onehot_encoded[:,0],'TD Y':onehot_encoded[:,1]})
    
    
    # In[99]:
    
    
    label_encoder7 = LabelEncoder()
    integer_encoded = label_encoder7.fit_transform(d1[:,15])
    onehot_encoder7 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder7.fit_transform(integer_encoded)
    ds6=pd.DataFrame({'CHF N':onehot_encoded[:,0],'CHF Y':onehot_encoded[:,1]})
    
    
    # In[100]:
    
    
    label_encoder8 = LabelEncoder()
    integer_encoded = label_encoder8.fit_transform(d1[:,16])
    onehot_encoder8 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder8.fit_transform(integer_encoded)
    ds7=pd.DataFrame({'DLP N':onehot_encoded[:,0],'DLP Y':onehot_encoded[:,1]})
    
    
    # In[101]:
    
    
    label_encoder9 = LabelEncoder()
    integer_encoded = label_encoder9.fit_transform(d1[:,20])
    onehot_encoder9 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder9.fit_transform(integer_encoded)
    ds8=pd.DataFrame({'WPP N':onehot_encoded[:,0],'WPP Y':onehot_encoded[:,1]})
    
    
    # In[102]:
    
    
    label_encoder10 = LabelEncoder()
    integer_encoded = label_encoder10.fit_transform(d1[:,21])
    onehot_encoder10 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder10.fit_transform(integer_encoded)
    ds9=pd.DataFrame({'Lung rales N':onehot_encoded[:,0],'Lung rales Y':onehot_encoded[:,1]})
    
    
    # In[103]:
    
    
    label_encoder11 = LabelEncoder()
    integer_encoded = label_encoder11.fit_transform(d1[:,22])
    onehot_encoder11 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder11.fit_transform(integer_encoded)
    ds10=pd.DataFrame({'SM N':onehot_encoded[:,0],'SM Y':onehot_encoded[:,1]})
    
    
    # In[104]:
    
    
    label_encoder12 = LabelEncoder()
    integer_encoded = label_encoder12.fit_transform(d1[:,23])
    onehot_encoder12 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder12.fit_transform(integer_encoded)
    ds11=pd.DataFrame({'DM N':onehot_encoded[:,0],'DM Y':onehot_encoded[:,1]})
    
    
    # In[105]:
    
    
    label_encoder13 = LabelEncoder()
    integer_encoded = label_encoder13.fit_transform(d1[:,25])
    onehot_encoder13 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder13.fit_transform(integer_encoded)
    ds12=pd.DataFrame({'Dyspnea N':onehot_encoded[:,0],'Dyspnea Y':onehot_encoded[:,1]})
    
    
    # In[106]:
    
    
    label_encoder14 = LabelEncoder()
    integer_encoded = label_encoder14.fit_transform(d1[:,27])
    onehot_encoder14 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder14.fit_transform(integer_encoded)
    ds13=pd.DataFrame({'Atypical N':onehot_encoded[:,0],'Atypical Y':onehot_encoded[:,1]})
    
    
    # In[107]:
    
    
    label_encoder15 = LabelEncoder()
    integer_encoded = label_encoder15.fit_transform(d1[:,28])
    onehot_encoder15 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder15.fit_transform(integer_encoded)
    ds14=pd.DataFrame({'Nonanginal N':onehot_encoded[:,0],'Nonanginal Y':onehot_encoded[:,1]})
    
    
    # In[108]:
    
    
    label_encoder16 = LabelEncoder()
    integer_encoded = label_encoder16.fit_transform(d1[:,30])
    onehot_encoder16= OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder16.fit_transform(integer_encoded)
    ds15=pd.DataFrame({'Exertional CP N':onehot_encoded[:,0],'Exertional CP Y':onehot_encoded[:,1]})
    
    
    # In[109]:
    
    
    label_encoder17 = LabelEncoder()
    integer_encoded = label_encoder17.fit_transform(d1[:,30])
    onehot_encoder17= OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder17.fit_transform(integer_encoded)
    ds16=pd.DataFrame({'LOwTh ang N':onehot_encoded[:,0],'LowTh angY':onehot_encoded[:,1]})
    
    
    # In[110]:
    
    
    label_encoder18 = LabelEncoder()
    integer_encoded = label_encoder18.fit_transform(d1[:,35])
    onehot_encoder18 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder18.fit_transform(integer_encoded)
    ds17=pd.DataFrame({'LVH N':onehot_encoded[:,0],'LVH Y':onehot_encoded[:,1]})
    
    
    # In[111]:
    
    
    label_encoder19 = LabelEncoder()
    integer_encoded = label_encoder19.fit_transform(d1[:,36])
    onehot_encoder19 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder19.fit_transform(integer_encoded)
    ds18=pd.DataFrame({'Poor R progression N':onehot_encoded[:,0],'Poor R Progression Y':onehot_encoded[:,1]})
    
    
    # In[112]:
    
    
    label_encoder20 = LabelEncoder()
    integer_encoded = label_encoder20.fit_transform(d1[:,37])
    onehot_encoder20 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder20.fit_transform(integer_encoded)
    ds19=pd.DataFrame({'LBBB':onehot_encoded[:,0],' BBB N':onehot_encoded[:,1],'RBBB':onehot_encoded[:,2]})
        
    
    
    # In[113]:
    
    
    label_encoder21 = LabelEncoder()
    integer_encoded = label_encoder21.fit_transform(d1[:,54])
    onehot_encoder21 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder21.fit_transform(integer_encoded)
    ds20=pd.DataFrame({'Moderate':onehot_encoded[:,0],'VHD N':onehot_encoded[:,1],'mild':onehot_encoded[:,3],'Severe':onehot_encoded[:,2]})
    
    
    # In[114]:
    
    
    lencod = LabelEncoder()
    label = lencod.fit_transform(d2[:,])
    #ds21=pd.DataFrame({'Cad':onehot_encoded[:,0],'Normal':onehot_encoded[:,1]})
    y = pd.DataFrame({'Cath':label[:,]})
    
    
    # In[115]:
    
    
    df1 = pd.concat([df,ds,ds1,ds2,ds3,ds4,ds5,ds6,ds7,ds8,ds9,ds10,ds11,ds12,ds13,ds14,ds15,ds16,ds17,ds18,ds19,ds20,y], axis =1)
    df1.head()
    
    
    # In[116]:
    
    
    df1.drop(["Sex","Obesity","CRF","CVA","Airway disease","Thyroid Disease","CHF","DLP","Weak Peripheral Pulse","Lung rales","Systolic Murmur","Diastolic Murmur","Dyspnea","Atypical","Nonanginal","Exertional CP","LowTH Ang","LVH","Poor R Progression","BBB","VHD"],axis=1, inplace = True)
    
    
    # In[117]:
    
    
    df3 =df1.copy()
    
    df2 = df1[["Age","Weight","Length","BMI","BP","PR","Function Class","FBS","TG","LDL","HDL","BUN","ESR","HB","K","Na","WBC","Lymph","Neut","PLT","EF-TTE","Region RWMA"]]
    df1 = df1.drop(["Age","Weight","Length","BMI","BP","PR","Function Class","FBS","TG","LDL","HDL","BUN","ESR","HB","K","Na","WBC","Lymph","Neut","PLT","EF-TTE","Region RWMA"], axis =1)
    df1.head()
    
    
    # In[118]:
    
    
    # In[119]:
    
    
    dfdata = pd.read_csv('final.csv')
    dfdata
    y=dfdata['Cath']
    dfdata=dfdata.drop(['Cath'],axis=1)
    y.head()
    dfdata.shape
    len(y)
    X_trainAndTest, X_validation, y_trainAndTest, y_validation = train_test_split(dfdata,y, test_size=0.20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_trainAndTest, y_trainAndTest, test_size=0.20, random_state=42)
    
    
    def crossover(col):
        a,b=random.randrange(0,80),random.randrange(0,80)
        for i in range(0,99):
            temp=i*2
            arr=[]
            arr=col[temp]
            for j in range(a,b):
                col[temp][j]=col[temp+1][j]
            for j in range(a,b):
                col[temp+1][j]=col[j]
        
    
    
    # In[121]:
    
    
    def mutation(col):
        for i in range(0,len(col)):   
            for j in range(0,16):
                a=random.randrange(0,80)
                b=random.randrange(0,100)
                if b<21:
                    if col[i][j]==1:
                        col[i][j]=0
                    if col[i][j]==0:
                        col[i][j]=1
    
    
    # In[122]:
    
    
    def selection(col):
        temp=[]
        for i in range(0,200):
            a,b=random.randrange(0,200),random.randrange(0,200)
            acc1,acc2=lst[a],lst[b]
            c=random.randrange(0,100)
            if c<21:
                if acc1<acc2:
                    temp.append(col[a])
                if acc2<acc1:
                    temp.append(col[b])
            if c>=21:
                if acc1>acc2:
                    temp.append(col[a])
                if acc2>acc1:
                    temp.append(col[b])
        for i in range(0,len(temp)):
            col[i]=temp[i]

    
    # In[123]:
    
    
    def calc_fitness(mat):
        for i in range(0,len(mat)):
            arr=[]
            arr.append(mat[i])
            df1 =dfdata
            #df2= X_test
            for j in range(0,len(arr)):
                if arr[j]==0:
                    df1.drop([df1.columns[j]],axis=1,inplace=True)
                    #df2.drop([df2.columns[j]],axis=1,inplace=True)
            x_tr, x_te, y_tr, y_te = train_test_split(df1,y, test_size=0.20)
            clf = svm.NuSVC(kernel='linear',nu=0.2272,degree=4,gamma=0.9184)
            clf.fit(x_tr,y_tr)
            predict=clf.predict(x_te)
            temp=accuracy_score(y_te,predict)
            temp1=f1_score(y_te,predict)
            temp=temp*100
            temp1=temp1*100
            lst.append(temp)
            lst1.append(temp1)
        
        
        
    
    
    # In[124]:
    
    
    col=[]
    
    for k in range (0,200):
        row=[]
        for i in range(0,80):
            j=random.choice([0,1])
            row.append(j)
        col.append(row)
    lst=[]
    lst1=[]
    calc_fitness(col)
    lst
    
    
    # In[125]:
    
    
    gen_acc_lst=[]
    gen_lst=[]
    gen_acc_lst1=[]
    gen_lst1=[]
    for i in range(0,20):
        selection(col)
        crossover(col)
        mutation(col)
        lst=[]
        lst1=[]
        calc_fitness(col)
        maxpos = lst.index(max(lst))
        gen_lst.append(col[maxpos])
        gen_acc_lst.append(max(lst))
        maxpos1 = lst1.index(max(lst1))
        gen_lst1.append(col[maxpos1])
        gen_acc_lst1.append(max(lst1))
        
    
    # In[126]:
    
    
    print(gen_acc_lst)
    print(gen_acc_lst1)
    ma=max(gen_acc_lst)
    mb=min(gen_acc_lst)
    mc=max(gen_acc_lst1)
    md=min(gen_acc_lst1)
    m1 = (ma+mb)/2
    m2 = (mc+md)/2
    plt.plot(gen_acc_lst)
    plt.plot(gen_acc_lst1)
    plt.show()
    plt.savefig("ge2.png")
    lbl.configure(text="the NuSVM accuracy and f1_score is: "+str(m1) + "," + str(m2))

 
def clicked1234():
    

    df = pd.read_csv("data.csv")
    df
    y = df['Cath']
    
    df =df.drop(["Cath"],axis =1 )
    df.head()
    data=df
    data1 = y
    
    d1=data.values
    d2=data1.values
    
    
    ohe= OneHotEncoder(categorical_features=[4])
    data = df
    values = array(data)
    print(values)
    # integer encode
    
    
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(d1[:,3])
    onehot_encoder1 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder1.fit_transform(integer_encoded)
    ds=pd.DataFrame({'Male':onehot_encoded[:,1],'Female':onehot_encoded[:,0]})
    
    
    
    label_encoder2 = LabelEncoder()
    integer_encoded = label_encoder2.fit_transform(d1[:,10])
    onehot_encoder2 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder2.fit_transform(integer_encoded)
    ds1=pd.DataFrame({'Obe N':onehot_encoded[:,0],'Obe Y':onehot_encoded[:,1]})
    
    
    label_encoder3 = LabelEncoder()
    integer_encoded = label_encoder3.fit_transform(d1[:,11])
    onehot_encoder3 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder3.fit_transform(integer_encoded)
    ds2=pd.DataFrame({'CRF N':onehot_encoded[:,0],'CRF Y':onehot_encoded[:,1]})
    
    
    
    label_encoder4 = LabelEncoder()
    integer_encoded = label_encoder4.fit_transform(d1[:,12])
    onehot_encoder4 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder4.fit_transform(integer_encoded)
    ds3=pd.DataFrame({'CVA N':onehot_encoded[:,0],'CVA Y':onehot_encoded[:,1]})
    
    label_encoder5 = LabelEncoder()
    integer_encoded = label_encoder5.fit_transform(d1[:,13])
    onehot_encoder5 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder5.fit_transform(integer_encoded)
    ds4=pd.DataFrame({'AID N':onehot_encoded[:,0],'AID Y':onehot_encoded[:,1]})
    
    
    # In[98]:
    
    
    label_encoder6= LabelEncoder()
    integer_encoded = label_encoder6.fit_transform(d1[:,14])
    onehot_encoder6= OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder6.fit_transform(integer_encoded)
    ds5=pd.DataFrame({'TD N':onehot_encoded[:,0],'TD Y':onehot_encoded[:,1]})
    
    
    # In[99]:
    
    
    label_encoder7 = LabelEncoder()
    integer_encoded = label_encoder7.fit_transform(d1[:,15])
    onehot_encoder7 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder7.fit_transform(integer_encoded)
    ds6=pd.DataFrame({'CHF N':onehot_encoded[:,0],'CHF Y':onehot_encoded[:,1]})
    
    
    # In[100]:
    
    
    label_encoder8 = LabelEncoder()
    integer_encoded = label_encoder8.fit_transform(d1[:,16])
    onehot_encoder8 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder8.fit_transform(integer_encoded)
    ds7=pd.DataFrame({'DLP N':onehot_encoded[:,0],'DLP Y':onehot_encoded[:,1]})
    
    
    # In[101]:
    
    
    label_encoder9 = LabelEncoder()
    integer_encoded = label_encoder9.fit_transform(d1[:,20])
    onehot_encoder9 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder9.fit_transform(integer_encoded)
    ds8=pd.DataFrame({'WPP N':onehot_encoded[:,0],'WPP Y':onehot_encoded[:,1]})
    
    
    # In[102]:
    
    
    label_encoder10 = LabelEncoder()
    integer_encoded = label_encoder10.fit_transform(d1[:,21])
    onehot_encoder10 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder10.fit_transform(integer_encoded)
    ds9=pd.DataFrame({'Lung rales N':onehot_encoded[:,0],'Lung rales Y':onehot_encoded[:,1]})
    
    
    # In[103]:
    
    
    label_encoder11 = LabelEncoder()
    integer_encoded = label_encoder11.fit_transform(d1[:,22])
    onehot_encoder11 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder11.fit_transform(integer_encoded)
    ds10=pd.DataFrame({'SM N':onehot_encoded[:,0],'SM Y':onehot_encoded[:,1]})
    
    
    # In[104]:
    
    
    label_encoder12 = LabelEncoder()
    integer_encoded = label_encoder12.fit_transform(d1[:,23])
    onehot_encoder12 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder12.fit_transform(integer_encoded)
    ds11=pd.DataFrame({'DM N':onehot_encoded[:,0],'DM Y':onehot_encoded[:,1]})
    
    
    # In[105]:
    
    
    label_encoder13 = LabelEncoder()
    integer_encoded = label_encoder13.fit_transform(d1[:,25])
    onehot_encoder13 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder13.fit_transform(integer_encoded)
    ds12=pd.DataFrame({'Dyspnea N':onehot_encoded[:,0],'Dyspnea Y':onehot_encoded[:,1]})
    
    
    # In[106]:
    
    
    label_encoder14 = LabelEncoder()
    integer_encoded = label_encoder14.fit_transform(d1[:,27])
    onehot_encoder14 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder14.fit_transform(integer_encoded)
    ds13=pd.DataFrame({'Atypical N':onehot_encoded[:,0],'Atypical Y':onehot_encoded[:,1]})
    
    
    # In[107]:
    
    
    label_encoder15 = LabelEncoder()
    integer_encoded = label_encoder15.fit_transform(d1[:,28])
    onehot_encoder15 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder15.fit_transform(integer_encoded)
    ds14=pd.DataFrame({'Nonanginal N':onehot_encoded[:,0],'Nonanginal Y':onehot_encoded[:,1]})
    
    
    # In[108]:
    
    
    label_encoder16 = LabelEncoder()
    integer_encoded = label_encoder16.fit_transform(d1[:,30])
    onehot_encoder16= OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder16.fit_transform(integer_encoded)
    ds15=pd.DataFrame({'Exertional CP N':onehot_encoded[:,0],'Exertional CP Y':onehot_encoded[:,1]})
    
    
    # In[109]:
    
    
    label_encoder17 = LabelEncoder()
    integer_encoded = label_encoder17.fit_transform(d1[:,30])
    onehot_encoder17= OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder17.fit_transform(integer_encoded)
    ds16=pd.DataFrame({'LOwTh ang N':onehot_encoded[:,0],'LowTh angY':onehot_encoded[:,1]})
    
    
    # In[110]:
    
    
    label_encoder18 = LabelEncoder()
    integer_encoded = label_encoder18.fit_transform(d1[:,35])
    onehot_encoder18 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder18.fit_transform(integer_encoded)
    ds17=pd.DataFrame({'LVH N':onehot_encoded[:,0],'LVH Y':onehot_encoded[:,1]})
    
    
    # In[111]:
    
    
    label_encoder19 = LabelEncoder()
    integer_encoded = label_encoder19.fit_transform(d1[:,36])
    onehot_encoder19 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder19.fit_transform(integer_encoded)
    ds18=pd.DataFrame({'Poor R progression N':onehot_encoded[:,0],'Poor R Progression Y':onehot_encoded[:,1]})
    
    
    # In[112]:
    
    
    label_encoder20 = LabelEncoder()
    integer_encoded = label_encoder20.fit_transform(d1[:,37])
    onehot_encoder20 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder20.fit_transform(integer_encoded)
    ds19=pd.DataFrame({'LBBB':onehot_encoded[:,0],' BBB N':onehot_encoded[:,1],'RBBB':onehot_encoded[:,2]})
        
    
    
    # In[113]:
    
    
    label_encoder21 = LabelEncoder()
    integer_encoded = label_encoder21.fit_transform(d1[:,54])
    onehot_encoder21 = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder21.fit_transform(integer_encoded)
    ds20=pd.DataFrame({'Moderate':onehot_encoded[:,0],'VHD N':onehot_encoded[:,1],'mild':onehot_encoded[:,3],'Severe':onehot_encoded[:,2]})
    
    
    # In[114]:
    
    
    lencod = LabelEncoder()
    label = lencod.fit_transform(d2[:,])
    #ds21=pd.DataFrame({'Cad':onehot_encoded[:,0],'Normal':onehot_encoded[:,1]})
    y = pd.DataFrame({'Cath':label[:,]})
    
    
    # In[115]:
    
    
    df1 = pd.concat([df,ds,ds1,ds2,ds3,ds4,ds5,ds6,ds7,ds8,ds9,ds10,ds11,ds12,ds13,ds14,ds15,ds16,ds17,ds18,ds19,ds20,y], axis =1)
    df1.head()
    
    
    # In[116]:
    
    
    df1.drop(["Sex","Obesity","CRF","CVA","Airway disease","Thyroid Disease","CHF","DLP","Weak Peripheral Pulse","Lung rales","Systolic Murmur","Diastolic Murmur","Dyspnea","Atypical","Nonanginal","Exertional CP","LowTH Ang","LVH","Poor R Progression","BBB","VHD"],axis=1, inplace = True)
    
    
    # In[117]:
    
    
    df3 =df1.copy()
    
    df2 = df1[["Age","Weight","Length","BMI","BP","PR","Function Class","FBS","TG","LDL","HDL","BUN","ESR","HB","K","Na","WBC","Lymph","Neut","PLT","EF-TTE","Region RWMA"]]
    df1 = df1.drop(["Age","Weight","Length","BMI","BP","PR","Function Class","FBS","TG","LDL","HDL","BUN","ESR","HB","K","Na","WBC","Lymph","Neut","PLT","EF-TTE","Region RWMA"], axis =1)
    df1.head()
    
    
    # In[118]:
    
    
    # In[119]:
    
    
    dfdata = pd.read_csv('final.csv')
    dfdata
    y=dfdata['Cath']
    dfdata=dfdata.drop(['Cath'],axis=1)
    y.head()
    dfdata.shape
    len(y)
    X_trainAndTest, X_validation, y_trainAndTest, y_validation = train_test_split(dfdata,y, test_size=0.20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_trainAndTest, y_trainAndTest, test_size=0.20, random_state=42)
    def crossover(col):
        a,b=random.randrange(0,80),random.randrange(0,80)
        for i in range(0,99):
            temp=i*2
            arr=[]
            arr=col[temp]
            for j in range(a,b):
                col[temp][j]=col[temp+1][j]
            for j in range(a,b):
                col[temp+1][j]=col[j]
        
    
    
    # In[121]:
    
    
    def mutation(col):
        for i in range(0,len(col)):   
            for j in range(0,16):
                a=random.randrange(0,80)
                b=random.randrange(0,100)
                if b<21:
                    if col[i][j]==1:
                        col[i][j]=0
                    if col[i][j]==0:
                        col[i][j]=1
    
    
    # In[122]:
    
    
    def selection(col):
        temp=[]
        for i in range(0,200):
            a,b=random.randrange(0,200),random.randrange(0,200)
            acc1,acc2=lst[a],lst[b]
            c=random.randrange(0,100)
            if c<21:
                if acc1<acc2:
                    temp.append(col[a])
                if acc2<acc1:
                    temp.append(col[b])
            if c>=21:
                if acc1>acc2:
                    temp.append(col[a])
                if acc2>acc1:
                    temp.append(col[b])
        for i in range(0,len(temp)):
            col[i]=temp[i]
    
    
    # In[123]:
    
        
    def calc_fitness(mat):
        for i in range(0,len(mat)):
            arr=[]
            arr.append(mat[i])
            df1 =X_train
            #df2= X_test
            for j in range(0,len(arr)):
                if arr[j]==0:
                    df1.drop([df1.columns[j]],axis=1,inplace=True)
                    #df2.drop([df2.columns[j]],axis=1,inplace=True)
            x_tr, x_te, y_tr, y_te = train_test_split(df1,y_train, test_size=0.20)
            clf = svm.LinearSVC( C = 1.0, loss='squared_hinge')
            clf.fit(x_tr,y_tr)
            predict=clf.predict(x_te)
            temp=accuracy_score(y_te,predict)
            temp1=f1_score(y_te,predict)
            temp=temp*100
            temp1=temp1*100
            lst.append(temp)
            lst1.append(temp1)
    
    
    
 
        
        
        
    
    
    # In[124]:
    
    
    col=[]
    
    for k in range (0,200):
        row=[]
        for i in range(0,80):
            j=random.choice([0,1])
            row.append(j)
        col.append(row)
    lst=[]
    lst1=[]
    calc_fitness(col)
    lst
    
    
    # In[125]:
    
    
    gen_acc_lst=[]
    gen_lst=[]
    gen_acc_lst1=[]
    gen_lst1=[]
    for i in range(0,20):
        selection(col)
        crossover(col)
        mutation(col)
        lst=[]
        lst1=[]
        calc_fitness(col)
        maxpos = lst.index(max(lst))
        gen_lst.append(col[maxpos])
        gen_acc_lst.append(max(lst))
        maxpos1 = lst1.index(max(lst1))
        gen_lst1.append(col[maxpos1])
        gen_acc_lst1.append(max(lst1))
        
    
    # In[126]:
    
    
    print(gen_acc_lst)
    print(gen_acc_lst1)
    ma=max(gen_acc_lst)
    mb=min(gen_acc_lst)
    mc=max(gen_acc_lst1)
    md=min(gen_acc_lst1)
    m1 = (ma+mb)/2
    m2 = (mc+md)/2
    plt.plot(gen_acc_lst)
    plt.plot(gen_acc_lst1)
    plt.savefig("ge3.png")
    lbl.configure(text="the LinearSVC accuracy and f1_score is: "+str(m1) + "," + str(m2))
"""
def grph():
    #import pp.py as pp
    #pp.pras()
    exec('file.py')
    #os.system("pp.py")
"""
btn = Button(window, text="Run SVC", command=clicked)
btn2 = Button(window, text="Run NuSVM", command=clicked12)
btn3 = Button(window, text="Run LinearSVC", command= clicked1234)
    
#btn = Button(window, text="Run me", command=clicked)
btn1=Button(window, text="neded graph?", command=grph)
 
btn.grid(column=0, row=2,pady=10)
btn1.grid(column=1,row=2,pady=10)
btn2.grid(column=0,row=4,pady=10)
btn3.grid(column=0, row=6,pady =10)
 
window.mainloop()