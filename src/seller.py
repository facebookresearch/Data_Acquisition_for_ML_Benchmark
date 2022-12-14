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

from pricefunction import PriceFunction
import numpy

class Seller(object):

    def __init__(self):
        return
        
    def loaddata(self, 
                 data=None, 
                 datapath=None,):
        # data: a m x n matrix
        # datapath: a path to a csv file.
        # the file should be a matrix with column names.
        if(not (data is None)):
            self.data = data
            return
        if(datapath != None):
            self.data = numpy.loadtxt(open(datapath, "rb"), 
                                      delimiter=",", 
                                      skiprows=1)
            return
        print("Not implemented load data of seller")
        return
    
    def setprice(self, pricefunc):
        self.pricefunc = pricefunc
        
    def getprice(self,data_size):
        q1 = data_size/(len(self.data))
        return self.pricefunc.get_price(q1) 
    
    def getdata(self, data_size, price):
        data = self.data
        q1 = data_size/(len(self.data))
        if(self.pricefunc.get_price(q1) <= price):
            number_of_rows = self.data.shape[0]
            random_indices = numpy.random.choice(number_of_rows, 
                                  size=data_size, 
                                  replace=False)
            rows = data[random_indices, :]
            return rows
            return 0
        else:
            raise ValueError("The buyer's offer is too small!")
        return
        

    
def main():
    print("test of the seller")
    MySeller = Seller()
    
    MySeller.loaddata(data=numpy.asmatrix([[0,1,1],[1,0,1]]))
    
    MyPricing = PriceFunction()
    MyPricing.setup(max_p = 100, method="lin")
    
    MySeller.setprice(MyPricing)
    
    data = MySeller.getdata(1,60)
    
    print("get data is ",data)
    
if __name__ == '__main__':
    main()        
    
