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


class MarketEngine(object):
    def __init__(self):
        return
    
    def setup_market(self, 
                     seller_data=None,
                     seller_prices=None,
                     buyer_data=None,
                     buyer_budget=None,
                     mlmodel=None):
        sellers = list()
        for i in range(len(seller_data)):
            MySeller = Seller()
            MySeller.loaddata(data=seller_data[i])
            MySeller.setprice(seller_prices[i])
            sellers.append(MySeller)
        self.sellers = sellers
        
        MyBuyer = Buyer()    
        MyBuyer.loaddata(data=buyer_data)     
        mlmodel1 = mlmodel
        MyBuyer.load_mlmodel(mlmodel1)
        self.buyer = MyBuyer
        self.buyer_budget = buyer_budget
        print("set up the market")
        return

    def load_stretagy(self,
                      stretagy=None,):
        self.stretagy = stretagy

        return
    
    def train_buyer_model(self):
        print(" train buyer model ")
        
        
        # check if the budget constraint is satisified.
        cost = sum(self.stretagy[1])
        if(cost>self.buyer_budget):
            raise ValueError("The budget constraint is not satisifed!")
            return
        
        traindata = None
        for i in range(len(self.sellers)):
            d1 = self.sellers[i].getdata(self.stretagy[0][i],self.stretagy[1][i])
            if(i==0):
                traindata = d1
            else:
                traindata = numpy.concatenate((traindata,d1))
            print(i,d1)

        print("budget checked! data loaded!")                
        #print("train data", traindata)   
        acc = self.buyer.train_mlmodel(traindata)    
        return acc
    
    
def main():
    print("test of the market engine")
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
    MyMarketEngine.load_stretagy(stretagy)
    
    acc1 = MyMarketEngine.train_buyer_model()
    print("acc is ",acc1)
    
    
if __name__ == '__main__':
    main()        
    
    
    
    
