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

import numpy
import pickle
import json
from pricefunction import PriceFunction
import pandas
class Dam(object):
    def __init__(self, 
                 instance=0,
                 ):
        self._instance = instance
        self._marketpath="../marketinfo/"
        if(instance not in [0,1,2,3,4]):
            raise ValueError("the instance id is incorrect. it must be 0, 1, 2, 3, or 4.")
        return 

    def getbudget(self,):
        budget = numpy.loadtxt(self._marketpath+str(self._instance)+"/price/"+"/budget.txt")
        return float(budget)
    
    def getbuyerdata(self,):
        path = self._marketpath+str(self._instance)+"/data_buyer/"+"/20.csv"
        buydata = pandas.read_csv(path,header=None,engine="pyarrow").to_numpy()
        return buydata
    
    def getmlmodel(self,):
        path = self._marketpath+str(self._instance)+"/data_buyer/"+"/mlmodel.pickle"
        with open(path, 'rb') as handle:
            model = pickle.load(handle)        
        return model

    def getsellerid(self,):
        path = self._marketpath+str(self._instance)+"/sellerid.txt"
        ids = numpy.loadtxt(path)
        return ids
    
    def getsellerinfo(self,seller_id):
        path = self._marketpath+str(self._instance)+"/summary/"+str(seller_id)+".csv.json"
        f = open(path)
        ids = json.load(f)
        
        price = numpy.loadtxt(self._marketpath+str(self._instance)+"/price/"+"/price.txt",
                                delimiter=',',dtype=str)
        price_i = price[seller_id]       
        MyPricing1 = PriceFunction()
        print("row number",ids['row_number'])
        MyPricing1.setup(max_p = float(price_i[1]), method=price_i[0], data_size=ids['row_number'])


        samples = numpy.loadtxt(self._marketpath+str(self._instance)+"/summary/"+str(seller_id)+".csvsamples.csv",
                                delimiter=' ',dtype=float)

                
        return MyPricing1, ids, samples


def main():    
    MyDam = Dam()
    budget = MyDam.getbudget() # get budget
    buyer_data = MyDam.getbuyerdata() # get buyer data
    mlmodel = MyDam.getmlmodel() # get ml model
    sellers_id = MyDam.getsellerid()
    i=0
    seller_i_price, seller_i_summary, seller_i_samples =  MyDam.getsellerinfo(seller_id=i)

    return 
    
if __name__ == "__main__":
   main()
