import numpy as np
import pandas as pd
import sys
import os
import matplotlib
class PLA:
    def __init__(self, X,y):
        
        self.X = X
        self.y = y 
        self.d, self.n = X.shape[1], X.shape[0]
        self.w = np.zeros(self.d)
        self.b = 1
        self.w_to_write = []
        
    def f(self, x):
        a = np.dot(self.w, x) +self.b
        sign = 1 if a > 0 else -1
        return sign

    def train(self):
        converge = False
        times = 0
        while True:
            steps = 0
            for i, x in enumerate(self.X):
                if self.y[i]*self.f(x) <= 0:
                    steps+=1
                    
                    self.w = self.w + self.y[i]*x
                    self.b = self.b + self.y[i]
                    w_to = list(self.w)
                    w_to.append(self.b)
                    self.w_to_write.append(w_to)
            if steps ==0:
                # converges
                break      
