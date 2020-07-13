#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
class Network():
    def __init__(self,layer_size,learning_rate,epoches=10000):
        self.layer_size=layer_size
        self.W=[np.random.randn(x,y) for x,y in zip(layer_size[:-1],layer_size[1:])]
        self.b=[np.random.randn(1,y) for y in layer_size[1:]]
        self.alpha=learning_rate
        self.epoches=epoches
        
    def SGD(self,data):
        """
        每次随机抽取全部样本的1/10训练
        """
        for epoch in range(self.epoches):
            np.random.shuffle(data)
            training_data=data[:max(1,len(data)//10)]
            x=np.array([i[0] for i in training_data])
            y=np.array([i[1] for i in training_data])
            self.fit_partial(x,y)
            
    def sigmoid(self,x):
        return 1.0/(1+np.exp(-x))
    
    def sigmoid_deriv(self,x):
        return x*(1-x)
        
    def fit_partial(self,x,y):
        A=[np.array(x)]
        for layer in range(len(self.W)):
            A.append(self.sigmoid(A[layer].dot(self.W[layer])+self.b[layer]))
        error=A[-1]-y
        D=[error*self.sigmoid_deriv(A[-1])]
        for layer in np.arange(len(A) - 2, 0, -1):
            D.append(D[-1].dot(self.W[layer].T)*self.sigmoid_deriv(A[layer]))
        D=D[::-1]
        for layer in np.arange(0, len(self.W)):
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])
            self.b[layer] += -self.alpha*np.ones((D[layer].shape[0],1)).T.dot(D[layer])
    
    def predict(self,x):
        A=[np.array(x)]
        for layer in range(len(self.W)):
            A.append(self.sigmoid(A[layer].dot(self.W[layer])+self.b[layer]))
        return A[-1]
    

