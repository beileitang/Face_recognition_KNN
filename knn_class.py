#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:55:13 2019

@author: beileitang
"""
import numpy as np


def calculate_eculidean_dis(xtrain,x):
    """
    # INPUT:
    # training dataset: array of image has been stored already 
    # predict : new img 
    #OUTPUT:
    the distance bewteen all traning img and new predict image 
    """
    dist = np.sum(np.square(xtrain- x))
    #print(np.dtype(dist))
    return (dist)


def cal_neighbours(train, labels, x, k, neighbor_loc=False):
    """
    # INPUT:
    # training dataset: array of image has been stored already 
    # labels: the name of training image
    # x : testing ONE image
    # k: k number of nearset neighbours 
    #OUTPUT:
    # neighbors: the k number of neighb
    """
#    img_dist = ()
    distances =[]
    for img in range(len(train)-1):
        img_dist = calculate_eculidean_dis(x,train[img])
        distances.append((train[img],img_dist,labels[img]))
        #print(distances)
    index=np.asarray(distances)
    neighbour_location = np.argsort(index[:,1])[:5]
    #print(index)
    
    distances_sort= sorted(distances, key=lambda j: j[1])[:k]
    #print(distances_sort)
    neighbors = distances_sort
    
    if neighbor_loc:
        return neighbors, neighbour_location
    else:
        return neighbors

# majorty vote to classifer the new example
def winner_vote(labels, display=True):
    """
    # INPUT:
    # neighbours: generated the K number of nearset nerighb 
    #OUTPUT:
    classfied results
    """
    from collections import Counter 
    list(labels)
    voted_label = labels
    temp = set(voted_label)
    result={}
    for i in temp:
        result[i]=voted_label.count(i) # 每个名字出现了的次数

    winner = [(k, result[k]) for k in sorted(result, key=result.get, reverse=True)]
#    for k, v in winner:
#        print(k, v)
    k = Counter(labels)
    high = k.most_common(1)
    if display:
        for k, v in winner:
            print(k, v)
        print("---------------------------- ")
        print("---------------------------- ")
        print("the picture is :")
        print(high)
        print("---------------------------- ")
        print("---------------------------- ")
    return high
    
    


