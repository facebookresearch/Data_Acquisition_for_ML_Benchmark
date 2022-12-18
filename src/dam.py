import numpy
import pickle
import json
from pricefunction import PriceFunction
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
        buydata = numpy.loadtxt(self._marketpath+str(self._instance)+"/data_buyer/"+"/20.csv",
                                delimiter=',')
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
        MyPricing1.setup(max_p = float(price_i[1]), method=price_i[0])


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