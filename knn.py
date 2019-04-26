import csv
import numpy as np
import distance
import operator
from pandas_ml import ConfusionMatrix

def printit(a):
    print(a[0][0],end=' ')
    print(a[0][1])
    print(a[1][0],end=' ')
    print(a[1][1])

def read_csv():
    a=[]
    with open("nursery.csv",'rt')as f:
        data=csv.reader(f)
        for row in data:
            a.append(row)
    return a

def trimdata(a,n):
    x=int((n*70)/100)
    training_data=[]
    test_data=[]
    for i in range(x):
        training_data.append(a[i])
    for i in range(x+1,n):
        test_data.append(a[i])
    return training_data,test_data

def preprocess(a):
    n=len(a)
    typec={0:{'usual':0,'pretentious':1,'great_pret':2},1:{'proper':0,'less_proper':1,'improper':2,'critical':3,'very_crit':4}}
    typec[2]={'complete':0,'completed':1,'incomplete':2,'foster':3}
    typec[3]={'1':0,'2':1,'3':2,'more':3}
    typec[4]={'convenient':0,'less_conv':1,'critical':2}
    typec[5]={'convenient':0,'inconv':1}
    typec[6]={'nonprob':0,'slightly_prob':1,'problematic':2}
    typec[7]={'recommended':0,'priority':1,'not_recom':2}
    for i in range(n-1):
        for j in range(8):
            x=typec[j]
            a[i][j]=x[a[i][j]]
    return a

def pairsort(a,b):
    a, b = (list(t) for t in zip(*sorted(zip(a,b))))
    return a,b
                
def knn(test,orig,k):
    index=[]
    dist=[]
    ct=1
    y=test[0:8]
    oc=test[8]
    for i in orig:
        x=i[0:8]
        dist.append(distance.levenshtein(test,x))
        index.append(i[8])
    
    dist,index=pairsort(dist,index)
    cp={}
    for i in dist:
        if ct<=k:
            
            if index[ct-1] in cp.keys():
                cp[index[ct-1]]+=1
            else:
                cp[index[ct-1]]=1
        else:
            break
        ct+=1
    predicted=max(cp.items(), key=operator.itemgetter(1))[0]
    return predicted,oc

filedata=read_csv()
n=len(filedata)
print(n)
n-=1
filedata=preprocess(filedata)
training_data,test_data=trimdata(filedata,n)
k=100
predicted=[]
actual=[]
for i in test_data:
    pre,oc=knn(i,training_data,k)
    predicted.append(pre)
    actual.append(oc)
cm = ConfusionMatrix(actual,predicted)
cm.print_stats()
    
            
