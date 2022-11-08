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
Created on Tue Aug 16 18:36:30 2022

@author: lingjiao
"""


from sklearn.linear_model import LogisticRegression



import numpy
from seller import Seller
from buyer import Buyer
from pricefunction import PriceFunction
from marketengine import MarketEngine

class Helper(object):
    def __init__(self):
        return
    def load_data(self, submission, MarketEngineObj):
        '''
        load submissions.
        return: train X and y 
        '''
        
        print(" train buyer model ")
        
        stretagy = submission
        buyer_budget = MarketEngineObj.buyer_budget
        
        # check if the budget constraint is satisified.
        cost = sum(stretagy[1])
        if(cost>buyer_budget):
            raise ValueError("The budget constraint is not satisifed!")
            return
        
        traindata = None
        for i in range(len(MarketEngineObj.sellers)):
            d1 = MarketEngineObj.sellers[i].getdata(stretagy[0][i],stretagy[1][i])
            if(i==0):
                traindata = d1
            else:
                traindata = numpy.concatenate((traindata,d1))
        return traindata
            
        
    def train_model(self, model, train_X, train_Y):
        model.fit(train_X,train_Y)
        return model 
    
    def eval_model(self, model, test_X, test_Y):
        eval_acc = model.score(test_X, test_Y)
        return eval_acc     
def main():
    print("test of the helper")
    MyMarketEngine = MarketEngine()
    
    data_1 = numpy.asmatrix([[0,1,0],[1,0,0]])               
    data_2 = numpy.asmatrix([[0,1,1],[1,0,1],[1,1,1],[0,0,1]])
    data_b = numpy.asmatrix([[0,1,0],[1,0,1],[0,1,1]])
                     
    buyer_budget = 100
           
    MyPricing1 = PriceFunction()
    MyPricing1.setup(max_p = 100, method="lin")
    MyPricing2 = PriceFunction()
    MyPricing2.setup(max_p = 100, method="lin")


    mlmodel1 = LogisticRegression(random_state=0)

             
    MyMarketEngine.setup_market(seller_data=[data_1,data_2],
                                seller_prices = [MyPricing1,MyPricing2],
                     buyer_data=data_b,
                     buyer_budget=buyer_budget,
                     mlmodel=mlmodel1,
                     )

    stretagy = [[1,2],[50,50]]
    #MyMarketEngine.load_stretagy(stretagy)
    
    #acc1 = MyMarketEngine.train_buyer_model()
    #print("acc is ",acc1)
    
    MyHelper = Helper()
    traindata = MyHelper.load_data(stretagy, MyMarketEngine)
    model = LogisticRegression(random_state=0)
    model = MyHelper.train_model(model, traindata[:,0:-1],
                                 numpy.ravel(traindata[:,-1]))
    acc1 = MyHelper.eval_model(model,test_X=data_b[:,0:-1],test_Y=data_b[:,-1])
    print("acc is:", acc1)
if __name__ == '__main__':
    main()        
    
    
    
    
