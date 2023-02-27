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

import matplotlib.pyplot as plt
import matplotlib

import numpy
from seller import Seller
from buyer import Buyer
from pricefunction import PriceFunction
from marketengine import MarketEngine
from helper import Helper
import pandas
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

def visualize_acc_cost(data_path="../logs/0/acc_cost_tradeoffs_uniform_logreg.csv",
                       savepath="../figures/",
                       ):
    plt.clf()
    data = pandas.read_csv(data_path)
    print("data",data)
    mean1 = data.groupby("budget").mean()
    var1 = data.groupby("budget").var()
    max1 = data.groupby("budget").max()
    min1 = data.groupby("budget").min()
    print("mean1",mean1['acc'])
    print("var",var1['acc'])
    print("diff, max, and min",max1['acc']-min1['acc'],max1['acc'],min1['acc'])
    sns.color_palette("tab10")
    swarm_plot  = sns.histplot(data=data, x="acc", hue="budget",palette=["C0", "C1", "C2","C3","C4"])
    #swarm_plot = sns.scatterplot(data=data, x= "cost",y="acc")
    plt.figure()
    fig = swarm_plot.get_figure()
    data_parse = data_path.split("/")
    method = data_parse[-1].split("_")[-2]
    instanceid = data_parse[-2]
    ml = data_parse[-1].split("_")[-1]    
    fig.savefig(savepath+str(instanceid)+"/"+method+ml+".pdf")

    plt.figure()

    swarm_plot  = sns.lineplot(data=data, y="acc", x="budget", err_style="band")
    fig2 = swarm_plot.get_figure()
    fig2.savefig(savepath+str(instanceid)+"/"+method+ml+"_line.pdf")


    return 

def evaluate(
        MarketHelper,
        MarketEngineObj,
        model,
        buyer_data,
        trial=100, # number of trials per budget
        seller_data_size_list = [100,200,300],
        cost_scale=0.1,
        method="single",
        ):
    trial_list = list(range(trial))
    acc_list = list()
    cost_list = list()    
    
    for i in range(trial):
        print("trial:",i)
        # generate a submission
        submission = gen_submission(seller_data_size_list,cost_scale=cost_scale,
                                    method=method)
        # calculate the cost of the submission
        cost = MarketHelper.get_cost(submission,MarketEngineObj)
        # generate the accuracy of the submission
        traindata = MarketHelper.load_data(submission, MarketEngineObj)
        model = MarketHelper.train_model(model, traindata[:,0:-1],
                                 numpy.ravel(traindata[:,-1]))
        acc1 = MarketHelper.eval_model(model,test_X=buyer_data[:,0:-1],test_Y=buyer_data[:,-1])

        cost_list.append(cost)
        acc_list.append(acc1)
    
    result = pandas.DataFrame()
    result['trial'] = trial_list
    result['acc'] = acc_list
    result['cost'] = cost_list
    return result

''' generate a pandas dataframe

trial,accuracy, cost
'''

def gen_submission(seller_data_size_list=[100,200,300],
                   cost_scale=1,
                   method="uniform"):
    if(method=="uniform"):
        submission = [numpy.random.randint(0,int(a*cost_scale)) for a in seller_data_size_list]
    if(method=="single"):
        submission = [0]*len(seller_data_size_list)        
        index = numpy.random.randint(0,len(submission))
        submission[index] = int(seller_data_size_list[index]*cost_scale)                               
    return submission

def evaluate_budget(MarketHelper,
        MarketEngineObj,
        model,
        buyer_data,
        trial=100, # number of trials per budget
        seller_data_size_list = [100,200,300],
        cost_scale_list=[0.1],
        method="single",
        ):
    results = [evaluate(
            MarketHelper=MarketHelper,
            MarketEngineObj=MarketEngineObj,
            model=model,
            buyer_data=buyer_data,
            trial=trial, # number of trials per budget
            seller_data_size_list = seller_data_size_list,
            cost_scale=c1,
            method=method,
            ) for c1 in cost_scale_list]
    full_result = pandas.concat(results, ignore_index=True,axis=0)
    return full_result

       
def main():
    matplotlib.pyplot.close('all')
    instance_ids = [0,1,2,3,4]
    methods = ['single','uniform']
    for instance_id in instance_ids:
        for method in methods:
            visualize_acc_cost(data_path="../logs/"+str(instance_id)+"/acc_cost_tradeoffs_"+method+"_knn.csv")
            visualize_acc_cost(data_path="../logs/"+str(instance_id)+"/acc_cost_tradeoffs_"+method+"_rf.csv")
            visualize_acc_cost(data_path="../logs/"+str(instance_id)+"/acc_cost_tradeoffs_"+method+"_logreg.csv")

    '''
    print("evaluate acc and cost tradeoffs")
    instance_id=0
    MyHelper = Helper()
    seller_data, seller_prices,  buyer_data, buyer_budget, data_size  = MyHelper.load_market_instance(
        feature_path="../features/"+str(instance_id)+"/",
        buyer_data_path="../marketinfo/"+str(instance_id)+"/data_buyer/20.csv",
        price_path="../marketinfo/"+str(instance_id)+"/price/price.txt",
        budget_path="../marketinfo/"+str(instance_id)+"/price/budget.txt",
        )
    
    MyMarketEngine = MarketEngine()
    mlmodel1 = LogisticRegression(random_state=0)
    mlmodel1 = KNeighborsClassifier(n_neighbors=9)	

    MyMarketEngine.setup_market(seller_data=seller_data,
                                seller_prices = seller_prices,
                                buyer_data=buyer_data,
                                buyer_budget=1e10,
                                mlmodel=mlmodel1,
                                )    

    result = evaluate(
        MarketHelper=MyHelper,
        MarketEngineObj=MyMarketEngine,
        model=mlmodel1,
        buyer_data=buyer_data,
        trial=10, # number of trials per budget
        seller_data_size_list = numpy.loadtxt("../marketinfo/"+str(instance_id)+"/seller_datasize.csv"),
        cost_scale=0.1,
        ) 
    result2 = evaluate_budget(
        MarketHelper=MyHelper,
        MarketEngineObj=MyMarketEngine,
        model=mlmodel1,
        buyer_data=buyer_data,
        trial=100, # number of trials per budget
        seller_data_size_list = numpy.loadtxt("../marketinfo/" + str(instance_id) +"/seller_datasize.csv"),
#        cost_scale_list=[0.005,0.0075,0.01,0.025],
#        method="uniform",
       cost_scale_list=[0.05,0.1,0.5,1],
       method="single",
        )
    folder1 = "../logs/"+str(instance_id)+"/"
    
    result2.to_csv(folder1+"acc_cost_tradeoffs.csv")
    print("result is:",result) 
    '''     
if __name__ == '__main__':
    main()        
    
    
    
    
