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
from helper import Helper

class Evaluator(object):
    def __init__(self):
        self.Helper = Helper()
        return
    def eval_submission(self, 
                        submission, 
                        seller_data,
                        buyer_data,
                        seller_price,
                        buyer_budget=100,
                        mlmodel=LogisticRegression(random_state=0),
                        ):
        '''
        

        Parameters
        ----------
        submission : TYPE
            DESCRIPTION.
        seller_data_path : TYPE
            DESCRIPTION.
        buyer_data_path : TYPE
            DESCRIPTION.
        price_data_path : TYPE
        mlmodel: TYPE
            DESCRIPTION.
         : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        ''' 
        
        MyMarketEngine = MarketEngine()
        MyHelper = self.Helper
    

        # set up the market
        MyMarketEngine.setup_market(seller_data=seller_data,
                                seller_prices = seller_price,
                     buyer_data=buyer_data,
                     buyer_budget=buyer_budget,
                     mlmodel=mlmodel,
                     )
        
        # get train data
        traindata = MyHelper.load_data(submission, MyMarketEngine)
        # train the model
        model = MyHelper.train_model(mlmodel, traindata[:,0:-1],
                                 numpy.ravel(traindata[:,-1]))
        # eval the model
        acc1 = MyHelper.eval_model(model,test_X=buyer_data[:,0:-1],test_Y=buyer_data[:,-1])   
        return acc1
       
def main():
    print("test of the evaluator")
    submission = [[1,2],[50,50]]
    data_1 = numpy.asmatrix([[0,1,0],[1,0,0]])               
    data_2 = numpy.asmatrix([[0,1,1],[1,0,1],[1,1,1],[0,0,1]])
    seller_data = [data_1, data_2]   
    buyer_data = numpy.asmatrix([[0,1,0],[1,0,1],[0,1,1]])
    MyPricing1 = PriceFunction()
    MyPricing1.setup(max_p = 100, method="lin")
    MyPricing2 = PriceFunction()
    MyPricing2.setup(max_p = 100, method="lin")
    seller_price = [MyPricing1, MyPricing2]
    
    MyEval = Evaluator()
    acc1 = MyEval.eval_submission( 
                        submission, 
                        seller_data,
                        buyer_data,
                        seller_price,
                        buyer_budget=100,
                        mlmodel=LogisticRegression(random_state=0),
                        )
    print("acc is:", acc1)
if __name__ == '__main__':
    main()        
    
    
    
    
