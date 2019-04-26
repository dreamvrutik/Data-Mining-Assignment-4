# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 12:55:31 2019

@author: Kunal
"""

#Imporing libraries

import random
import math
import statistics

#Constructing the dataset
def printit(a):
    print(a[0][0],end=' ')
    print(a[0][1])
    print(a[1][0],end=' ')
    print(a[1][1])


dataset = []

file = open("nursery.data","r")
for line in file.readlines():
    clean_line = line[:-1]
    curr_entry = clean_line.split(',')
    dataset.append(curr_entry)
file.close()

dataset = dataset[:-1]


#Discretizing the attributes

mapping = [{} for i in range(0,9)]
class_mapping = {}

for i in range(0,len(dataset)):
    for j in range(0,9):
        if dataset[i][j] not in mapping[j]:
            val = len(mapping[j])
            mapping[j][dataset[i][j]] = val
            if j == 8:
                class_mapping[val] = dataset[i][j]
            dataset[i][j] = val
        else:
            dataset[i][j] = mapping[j][dataset[i][j]]
            
#Splitting data into training and test set
            
random.shuffle(dataset)
val = math.ceil(0.7*(len(dataset)))
training_set = dataset[:val]
test_set = dataset[val:]


#Calculating summaries

    #Classifying training set
    
seperated_by_class = {}
for vector in training_set:
    if vector[-1] not in seperated_by_class:
        seperated_by_class[vector[-1]] = []
    seperated_by_class[vector[-1]].append(vector[:-1])
    
    
    #Calculating summaries for each class

summaries = {}
for i in seperated_by_class.keys():
    summaries[i] = []
    matrix = seperated_by_class[i]
    for j in range(0,8):
        attribute_values = []
        for k in range(0,len(matrix)):
            attribute_values.append(matrix[k][j])
        mean = statistics.mean(attribute_values)
        dev = statistics.stdev(attribute_values,xbar=None)
        temp = []
        temp.append(mean)
        temp.append(dev)
        summaries[i].append(temp)


#Using gaussian probability distribution constructed on each attribute to predict
        
def predict(record):
    global summaries
    maxim = 0
    ans = -1
    for cl in summaries.keys():
        summaries_cl = summaries[cl]
        prod = 1
        for i in range(0,8):
            mn = summaries_cl[i][0]
            dev = summaries_cl[i][1]
            val = record[i]
            prob = 1
            if dev == 0:
                if val != mn:
                    prob = 0
            else:
                exponent = math.exp(-(math.pow(val-mn,2)/(2*math.pow(dev,2))))
                prob = prob*((1/(math.sqrt(2*math.pi) * dev))*exponent)
            prod = prod*prob
        if prod>maxim:
            maxim = prod
            ans = cl
    return ans

#Predicting the test set
    
answers = []
correct = 0
tot = 0

for record in test_set:
    ans = record[-1]
    tot = tot+1
    record = record[:-1]
    pred = predict(record)
    if pred == ans:
        correct = correct+1
    answers.append(class_mapping[pred])

accuracy = correct/tot

actual_answers = []
for record in test_set:
    ans = record[-1]
    actual_answers.append(class_mapping[ans])
si=len(answers)
print(si)
not_recom=[[0,0],[0,0]]
priority=[[0,0],[0,0]]
recommend=[[0,0],[0,0]]
spec_prior=[[0,0],[0,0]]
very_recom=[[0,0],[0,0]]
for i in range(si):
    predicted=answers[i]
    oc=actual_answers[i]
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


        
    

