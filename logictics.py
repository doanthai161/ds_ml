import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('data_//data_classification.csv', header=None)
# print(data)
# print(data.values)
true_x=[]
true_y=[]
false_x=[]
false_y=[]

for item in data.values:
    if item[2]==1:
        true_x.append(item[0])
        true_y.append(item[1])
    else:
        false_x.append(item[0])
        false_y.append(item[1])
    
plt.scatter(true_x, true_y,marker='o',c='b')
plt.scatter(false_x, false_y,marker='o',c='r')
plt.show()

def sigmoid(z):
    return 1.0/(1+ np.exp(-z))

def phan_chia(p):
    if p >= 0.5:
        return 1
    else:
        return 0

def predict(feature, weight):
    z = np.dot(feature,weight)
    return sigmoid(z)

def cost_funtion(features, lables, weights):
    """
    :param features: (100x3)
    :param lables: (100x1)
    :param weight: (3x1)
    :return: chi phi cost
    """
    n =len(lables)
    prediction = predict(features,weights)
    """
    predictions
    [0.6, 0.7, 0.5,0.4]
    [0, 1, 0,0]
    ma trận chuyển vị
    [[0]
    [1]
    [0]
    [0]
    ]
    """
    cost_class1 = -lables*np.log(prediction)
    cost_class2 = -(1-lables)*np.log(1 - prediction)
    cost = cost_class1 + cost_class2
    return cost.sum()/n

def update_weight(features, lables, weights, learning_rate):
    """
    param feature: 100x3
    param lable:   100x1
    param weight:  3x1
    param learning_rate: float
    return: new weight: float
    """
    n = len(lables)
    prediction = predict(features, weights)
    gd = np.dot(features.T,(prediction - lables))
    gd = gd/n
    gd = gd*learning_rate
    weights = weights - gd
    return weights

