#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 23:14:33 2020

@author: jaideep
"""


import numpy as np
import pandas as pd
import pickle

def import_data():
    train_X = np.genfromtxt('train_X_nb.csv', dtype = 'str', delimiter = '\n')
    train_Y = np.genfromtxt('train_Y_nb.csv', dtype = int, delimiter = ' ')
    return train_X, train_Y

def preprocessing(X):
    def preprocessing(s):
        import string
        allowed = set(string.ascii_letters + ' ')
        s = ''.join(i for i in s if i in allowed)
        return ' '.join(s.split())   
    for data in X:
        data = preprocessing(data)
    return X

def class_wise_words_frequency_dict(X, Y):
    #TODO Complete the function implementation. Read the Question text for details
    classes = {}
    for i in range(len(X)):
        for j in set(Y):
            if Y[i] == int(j):
                if j not in classes:
                    classes[j] = {}
                    for word in X[i].split():
                        if word in classes[j]:
                            classes[j][word] += 1
                        else:
                            classes[j][word] = 1
                else:
                    for word in X[i].split():
                        if word in classes[j]:
                            classes[j][word] += 1
                        else:
                            classes[j][word] = 1
    return classes

def compute_prior_probabilities(Y):
    total = len(Y)
    prior_probabilities = {}
    for i in Y:
        for j in set(Y):
            if i == j:
                if j in prior_probabilities:
                    prior_probabilities[j] += 1
                else:
                    prior_probabilities[j] = 1
    for i in prior_probabilities:
        prior_probabilities[i] /= total
        prior_probabilities[i] = np.round(prior_probabilities[i], 3)
    return prior_probabilities

def get_class_wise_denominators_likelihood(X, Y):
    #TODO Complete the function implementation. Read the Question text for details
    vocab = " ".join(X)
    classes = set(Y)
    V = len(set(vocab.split()))
    class_wise_frequency_dict = class_wise_words_frequency_dict(X, Y)
    denominator_likelihood = {}
    for c in classes:
        denominator_likelihood[c] = V
        for word in class_wise_frequency_dict[c]:
            denominator_likelihood[c] += class_wise_frequency_dict[c][word]
    return denominator_likelihood  

def compute_likelihood(test_X, c):
    
    #TODO Complete the function implementation. Read the Question text for details
    class_wise_frequency_dict, class_wise_denominators, prior_probabilities = train(X, Y)
    vocab = [[word for word in class_wise_frequency_dict[words]]
            for words in class_wise_frequency_dict.keys()]
    vocalbulary = []
    for i in range(len(vocab)):
        vocalbulary += vocab[i]
    vocalbulary = list(set(vocalbulary))
    likelihood = 0
    for word in test_X.split():
        if word in vocalbulary:
            if word in class_wise_frequency_dict[c]:
                likelihood += np.log((class_wise_frequency_dict[c][word] + 1)/\
                class_wise_denominators[c])
            else:
                likelihood -= np.log(class_wise_denominators[c])
        else:
            likelihood -= np.log(class_wise_denominators[c])
    return likelihood

def train(X,Y):
    X = preprocessing(X)
    class_wise_frequency_dict = class_wise_words_frequency_dict(X, Y)
    class_wise_denominators = get_class_wise_denominators_likelihood(X, Y)
    prior_probabilities = compute_prior_probabilities(Y)
    return class_wise_frequency_dict, class_wise_denominators, prior_probabilities
    

def predict_test_case(test_X):
    predicted_class = {}
    class_wise_frequency_dict, class_wise_denominators, prior_probabilities = train(X, Y)
    for classes in class_wise_denominators.keys():
        predicted_class[classes] = np.log(prior_probabilities[classes])\
        + compute_likelihood(test_X, classes, class_wise_frequency_dict, class_wise_denominators)
    predicted_class = sorted(predicted_class.items(), key = lambda x: -x[1])
    return predicted_class[0][0]


    
    

if __name__ == '__main__':
    X, Y = import_data()
    class_wise_frequency_dict, class_wise_denominators, prior_probabilities = train(X,Y)
    model = [class_wise_frequency_dict, class_wise_denominators, prior_probabilities]
    with open('weights.pkl', 'wb') as f:
        pickle.dump(model, f)