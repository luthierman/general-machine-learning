import numpy as np
import pandas as pd
import sys
import os
import matplotlib
class LR:
    def __init__(self, X, y, alpha, num_iters ):
        self.X = X
        self.y = y 
        self.d, self.n = X.shape[1], X.shape[0]

        self.bias = 1
        self.b_age = 0
        self.b_weight = 0

        self.num_iters = num_iters
        self.alpha = alpha 
    def fx(self):
        return self.bias + self.b_age * self.X[:,0] + self.b_weight*self.X[:,1]
    def grad_desc(self):
        for e in range(self.num_iters):
            y_hat = self.fx()
            loss = y_hat - self.y
            grad_bias = self.bias - self.alpha *(1/self.n)*np.sum(loss @ np.ones(self.n))
            grad_age = self.b_age - self.alpha *(1/self.n)*np.sum(loss @ self.X[:,0])
            grad_weight = self.b_weight - self.alpha *(1/self.n)*np.sum(loss@ self.X[:,1])
            R = np.sum(loss**2)/(2*self.n)
            self.bias, self.b_age, self.b_weight = grad_bias, grad_age, grad_weight
        return [self.bias, self.b_age, self.b_weight]
        
        
