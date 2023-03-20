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
from sklearn.ensemble import GradientBoostingClassifier
import numpy
from marketengine import MarketEngine
from helper import Helper
from sklearn.neighbors import KNeighborsClassifier
import json

def evaluate_batch(data_config,
                   ):
    instance_ids = data_config['instance_ids']
    result = dict()
    for id1 in instance_ids:
        result[id1] = evaluate_multiple_trial(data_config,instance_id=id1)    
    return result

def evaluate_multiple_trial(data_config,
                            instance_id,
                            num_trial=10,
                            ):
    
    results = [evaluate_once(data_config=data_config,
                  instance_id=instance_id) for i in range(num_trial)]
    #print("results are:",results)
    results_avg = dict()
    results_avg['cost'] = 0
    results_avg['acc'] = 0
    for item in results:
        #print("item is:",item)
        results_avg['cost'] += item['cost']/len(results)
        results_avg['acc'] += item['acc']/len(results)
    return results_avg

def evaluate_once(data_config,
                  instance_id):
    # load submission
    submission = load_submission(path = data_config['submission_path']+str(instance_id)+".csv")
    
    # get the helper
    model_name = data_config['model_name']
    MarketHelper, MarketEngineObj, model, traindata, buyer_data = get_market_info(instance_id=instance_id,
                                                                                  model_name=model_name)
    
    # calculate the cost of the submission
    cost = MarketHelper.get_cost(submission,MarketEngineObj)
    
    # generate the accuracy of the submission
    traindata = MarketHelper.load_data(submission, MarketEngineObj)
    model = MarketHelper.train_model(model, traindata[:,0:-1],
                                 numpy.ravel(traindata[:,-1]))
    acc1 = MarketHelper.eval_model(model,test_X=buyer_data[:,0:-1],test_Y=buyer_data[:,-1])
    
    result = dict()
    result['cost'] = cost
    result['acc'] = acc1
    return result

def load_submission(path):
    data = numpy.loadtxt(path,delimiter=",",dtype=int)
    return data

def get_market_info(instance_id,
                    model_name="lr"):
    MyHelper = Helper()
    seller_data, seller_prices,  buyer_data, buyer_budget, data_size  = MyHelper.load_market_instance(
        feature_path="../features/"+str(instance_id)+"/",
        buyer_data_path="../marketinfo/"+str(instance_id)+"/data_buyer/20.csv",
        price_path="../marketinfo/"+str(instance_id)+"/price/price.txt",
        budget_path="../marketinfo/"+str(instance_id)+"/price/budget.txt",
        )
    MyMarketEngine = MarketEngine()
    mlmodel1 = LogisticRegression(random_state=0)
    if(model_name=="knn"):
        mlmodel1 = KNeighborsClassifier(n_neighbors=9)	
    if(model_name=='rf'):
        mlmodel1 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                   max_depth=1, random_state=0)
    MyMarketEngine.setup_market(seller_data=seller_data,
                                seller_prices = seller_prices,
                                buyer_data=buyer_data,
                                buyer_budget=1e10,
                                mlmodel=mlmodel1,
                                ) 
    
    return MyHelper, MyMarketEngine, mlmodel1,seller_data, buyer_data

def main():
    data_config = json.load(open("../config/bilge20230301_rf.json")) # load the data folder
    result = evaluate_batch(data_config)
    json_object = json.dumps(result, indent=4)
    save_path = data_config['save_path']
    with open(save_path, "w") as outfile:
        outfile.write(json_object)
    print("The result is:",result)
    
    return
 
if __name__ == '__main__':
    main()        
    
    
    
    
