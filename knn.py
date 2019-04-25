import csv
import numpy as np
import distance
import operator
from sklearn.metrics import confusion_matrix as cm

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
not_recom=[[0,0],[0,0]]
priority=[[0,0],[0,0]]
recommend=[[0,0],[0,0]]
spec_prior=[[0,0],[0,0]]
very_recom=[[0,0],[0,0]]
for i in test_data:
    predicted,oc=knn(i,training_data,k)
    if oc=='not_recom':
        if predicted==oc:
            not_recom[1][1]+=1
            priority[0][0]+=1
            recommend[0][0]+=1
            spec_prior[0][0]+=1
            very_recom[0][0]+=1
        else:
            not_recom[1][0]+=1
            if predicted=='priority':
                priority[0][1]+=1
                recommend[0][0]+=1
                spec_prior[0][0]+=1
                very_recom[0][0]+=1
            elif predicted=='recommend':
                recommend[0][1]+=1
                priority[0][0]+=1
                spec_prior[0][0]+=1
                very_recom[0][0]+=1
            elif predicted=='spec_prior':
                spec_prior[0][1]+=1
                priority[0][0]+=1
                recommend[0][0]+=1
                very_recom[0][0]+=1
            elif predicted=='very_recom':
                very_recom[0][1]+=1
                priority[0][0]+=1
                recommend[0][0]+=1
                spec_prior[0][0]+=1
    elif oc=='priority':
        if predicted==oc:
            priority[1][1]+=1
            not_recom[0][0]+=1
            recommend[0][0]+=1
            spec_prior[0][0]+=1
            very_recom[0][0]+=1
        else:
            priority[1][0]+=1
            if predicted=='not_recom':
                not_recom[0][1]+=1
                recommend[0][0]+=1
                spec_prior[0][0]+=1
                very_recom[0][0]+=1
            elif predicted=='recommend':
                recommend[0][1]+=1
                not_recom[0][0]+=1
                spec_prior[0][0]+=1
                very_recom[0][0]+=1
            elif predicted=='spec_prior':
                spec_prior[0][1]+=1
                not_recom[0][0]+=1
                recommend[0][0]+=1
                very_recom[0][0]+=1
            elif predicted=='very_recom':
                very_recom[0][1]+=1
                not_recom[0][0]+=1
                recommend[0][0]+=1
                spec_prior[0][0]+=1
    elif oc=='recommend':
        if predicted==oc:
            recommend[1][1]+=1
            priority[0][0]+=1
            not_recom[0][0]+=1
            spec_prior[0][0]+=1
            very_recom[0][0]+=1
        else:
            recommend[1][0]+=1
            if predicted=='priority':
                priority[0][1]+=1
                not_recom[0][0]+=1
                spec_prior[0][0]+=1
                very_recom[0][0]+=1
            elif predicted=='not_recom':
                not_recom[0][1]+=1
                priority[0][0]+=1
                spec_prior[0][0]+=1
                very_recom[0][0]+=1
            elif predicted=='spec_prior':
                spec_prior[0][1]+=1
                priority[0][0]+=1
                not_recom[0][0]+=1
                very_recom[0][0]+=1
            elif predicted=='very_recom':
                very_recom[0][1]+=1
                priority[0][0]+=1
                not_recom[0][0]+=1
                spec_prior[0][0]+=1
    elif oc=='spec_prior':
        if predicted==oc:
            spec_prior[1][1]+=1
            priority[0][0]+=1
            not_recom[0][0]+=1
            recommend[0][0]+=1
            very_recom[0][0]+=1
        else:
            spec_prior[1][0]+=1
            if predicted=='priority':
                priority[0][1]+=1
                not_recom[0][0]+=1
                recommend[0][0]+=1
                very_recom[0][0]+=1
            elif predicted=='not_recom':
                not_recom[0][1]+=1
                priority[0][0]+=1
                recommend[0][0]+=1
                very_recom[0][0]+=1
            elif predicted=='recommend':
                recommend[0][1]+=1
                priority[0][0]+=1
                not_recom[0][0]+=1
                very_recom[0][0]+=1
            elif predicted=='very_recom':
                very_recom[0][1]+=1
                priority[0][0]+=1
                not_recom[0][0]+=1
                recommend[0][0]+=1
    elif oc=='very_recom':
        if predicted==oc:
            very_recom[1][1]+=1
            priority[0][0]+=1
            not_recom[0][0]+=1
            spec_prior[0][0]+=1
            recommend[0][0]+=1
        else:
            very_recom[1][0]+=1
            if predicted=='priority':
                priority[0][1]+=1
                not_recom[0][0]+=1
                spec_prior[0][0]+=1
                recommend[0][0]+=1
            elif predicted=='not_recom':
                not_recom[0][1]+=1
                priority[0][0]+=1
                spec_prior[0][0]+=1
                recommend[0][0]+=1
            elif predicted=='spec_prior':
                spec_prior[0][1]+=1
                priority[0][0]+=1
                not_recom[0][0]+=1
                recommend[0][0]+=1
            elif predicted=='recommend':
                recommend[0][1]+=1
                priority[0][0]+=1
                not_recom[0][0]+=1
                spec_prior[0][0]+=1


print("*****FOR CLASS : not_recom*****")
printit(not_recom)
accuracy=not_recom[0][0]+not_recom[1][1]
tsum=not_recom[0][0]+not_recom[0][1]+not_recom[1][0]+not_recom[1][1]
if(tsum==0):
    print("Divide by zero .")
else:
    accuracy/=tsum
    print("Accuracy = ",accuracy)
tp=not_recom[1][1]
fn=not_recom[0][1]
recall=tp/(tp+fn)
if(tp+fn==0):
    print("Divide by 0")
else:
    print("Recall = ",recall)
if(not_recom[1][1]+not_recom[0][1]==0):
    print("Divide by zero")
else :
    precision=not_recom[1][1]/(not_recom[1][1]+not_recom[0][1])
    print("Precision = ",precision)
print("*****FOR CLASS : priority*****")
printit(priority)
accuracy=priority[0][0]+priority[1][1]
tsum=priority[0][0]+priority[0][1]+priority[1][0]+priority[1][1]
if tsum!=0:
    accuracy/=tsum
    print("Accuracy = ",accuracy)
else:
    print("Divide by zero")
tp=priority[1][1]
fn=priority[0][1]
if tp+fn!=0:
    recall=tp/(tp+fn)
    print("Recall = ",recall)
else:
    print("Divide by 0")
if priority[1][1]+priority[0][1]!=0:
    precision=priority[1][1]/(priority[1][1]+priority[0][1])
    print("Precision = ",precision)
else:
    print("Divide by 0")
print("*****FOR CLASS : recommend*****")
printit(recommend)
accuracy=recommend[0][0]+recommend[1][1]
tsum=recommend[0][0]+recommend[0][1]+recommend[1][0]+recommend[1][1]
if tsum!=0:
    accuracy/=tsum
    print("Accuracy = ",accuracy)
else:
    print("Divide by zero")
tp=recommend[1][1]
fn=recommend[0][1]
if tp+fn!=0:
    recall=tp/(tp+fn)
    print("Recall = ",recall)
else:
    print("Divide by 0")
if recommend[1][1]+recommend[0][1]!=0:
    precision=recommend[1][1]/(recommend[1][1]+recommend[0][1])
    print("Precision = ",precision)
else:
    print("Divide by 0")
    
print("*****FOR CLASS : spec_prior*****")
printit(spec_prior)
accuracy=spec_prior[0][0]+spec_prior[1][1]
tsum=spec_prior[0][0]+spec_prior[0][1]+spec_prior[1][0]+spec_prior[1][1]
if tsum!=0:
    accuracy/=tsum
    print("Accuracy = ",accuracy)
else:
    print("Divide by zero")
tp=spec_prior[1][1]
fn=spec_prior[0][1]
if tp+fn!=0:
    recall=tp/(tp+fn)
    print("Recall = ",recall)
else:
    print("Divide by 0")
if spec_prior[1][1]+spec_prior[0][1]!=0:    
    precision=spec_prior[1][1]/(spec_prior[1][1]+spec_prior[0][1])
    print("Precision = ",precision)
else:
    print("Divide by 0")
print("*****FOR CLASS : very_recom*****")
printit(very_recom)
accuracy=very_recom[0][0]+very_recom[1][1]
tsum=very_recom[0][0]+very_recom[0][1]+very_recom[1][0]+very_recom[1][1]
if tsum!=0:
    accuracy/=tsum
    print("Accuracy = ",accuracy)
else:
    print("Divide by zero")
tp=very_recom[1][1]
fn=very_recom[0][1]
if tp+fn!=0:
    recall=tp/(tp+fn)
    print("Recall = ",recall)
else:
    print("Divide by 0")
if very_recom[1][1]+very_recom[0][1]!=0:
    precision=very_recom[1][1]/(very_recom[1][1]+very_recom[0][1])
    print("Precision = ",precision)
else:
    print("Divide by 0")


        
    
