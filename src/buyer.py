"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 18:35:29 2022

@author: lingjiao
"""

from sklearn.linear_model import LogisticRegression

import numpy

class Buyer(object):

    def __init__(self):
        return
        
    def loaddata(self, 
                 data=None, 
                 datapath=None,):
        if(not (data is None)):
            self.data = data
            return
        if(datapath != None):
            self.data = numpy.loadtxt(open(datapath, "rb"), 
                                      delimiter=",", 
                                      skiprows=1)
            return
        raise ValueError("Not implemented load data of buyer")
        return
    
    def load_stretagy(self, 
                      stretagy=None):
        return
    
    def get_stretagy(self):
        return self.stretagy

    def load_mlmodel(self,
                     mlmodel):
        self.mlmodel = mlmodel
        return 0

    def train_mlmodel(self,
                      train_data):
        
        X = train_data[:,0:-1]
        y = numpy.ravel(train_data[:,-1])
        self.mlmodel.fit(X,y)
        X_1 = self.data[:,0:-1]
        y_1 = numpy.ravel(self.data[:,-1])
        eval_acc = self.mlmodel.score(X_1, y_1)
        return eval_acc
    
        
def main():
    print("test of the buyer")
    MyBuyer = Buyer()
    
    
    
    MyBuyer.loaddata(data=numpy.asmatrix([[0,1,1,1],[1,0,1,0]]))
    
    mlmodel1 = LogisticRegression(random_state=0)
    
    MyBuyer.load_mlmodel(mlmodel1)

    train_data = numpy.asmatrix([[0,1,1,1],[1,0,1,0],[1,1,1,1]])
    
    eval1 = MyBuyer.train_mlmodel(train_data)    
    
    print("eval acc",eval1)

if __name__ == '__main__':
    main()        
