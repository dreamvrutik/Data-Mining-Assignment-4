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
        dev = statistics.stdev(attribute_values)
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
