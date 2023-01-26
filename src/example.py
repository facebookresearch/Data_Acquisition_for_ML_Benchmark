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

from dam import Dam
print("Loading Dataset...")
instance=2 # instance id, can be 0,1,2,3,4
MyDam = Dam(instance=instance)
print("Dataset loaded!")
budget = MyDam.getbudget() # get budget
print("budget is:",budget)
# 3. Display seller_data 
buyer_data = MyDam.getbuyerdata() # get buyer data
print("buyer data is:",buyer_data)
mlmodel = MyDam.getmlmodel() # get ml model
print("mlmodel is",mlmodel)
sellers_id = MyDam.getsellerid() # seller ids
print("seller ids are", sellers_id)
for i in sellers_id:
    seller_i_price, seller_i_summary, seller_i_samples = MyDam.getsellerinfo(seller_id=int(i))
    print("seller ", i, " price: ", seller_i_price)
    print("seller ", i, " summary: ", seller_i_summary)
    print("seller ", i, " samples: ", seller_i_samples)
	
