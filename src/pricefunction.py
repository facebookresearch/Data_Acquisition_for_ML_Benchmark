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



class PriceFunction(object):

    def __init__(self):
        return
        
    def setup(self, max_p = 100, method="lin",
            data_size=1):
        self.max_p = max_p
        self.method = "lin"
        self.data_size = data_size

    def get_price(self, 
                 frac=1, 
                 ):
        if(frac<0 or frac>1):
            raise ValueError("The fraction of samples must be within [0,1]!")
        max_p = self.max_p
        if(self.method=="lin"):
            p1 = max_p * frac
            return p1
        
        return

    def get_price_samplesize(self,
                            samplesize=10,
                            ):
        frac = samplesize/self.data_size
        #print("frac is",frac)
        return self.get_price(frac)

        

    
def main():
    print("test of the price func")
    
    
if __name__ == '__main__':
    main()        
    
