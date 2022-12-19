# dataperf-dam: A Data-centric Benchmark on Data Acquisition for Machine Learning

This github repo serves as the starting point for submissions and evaluations for data acquisition for machine learning benchmark, or in short, DAM, as part of the DataPerf benchmark suite [https://dataperf.org/](https://dataperf.org/)


## 1. What is the DAM benchmark?

An increasingly large amount of data is purchased for AI-enabled data science applications. How to select the right set of datasets for AI tasks of interest is an important decision that has, however, received limited attention. A naive approach is to acquire all available datasets and then select which ones to use empirically. This requires expensive human supervision and incurs prohibitively high costs, posing unique challenges to budget-limited users. 

How can one decide which datasets to acquire before actually purchasing the data to optimize the performance quality of an ML model?  In the DAM (Data-Acquisition-for-Machine-learning) benchmark, the participants are asked to tackle the aforementioned problem. Participants need to provide a data purchase strategy for a data buyer in K (=5 in the beta version) separate data marketplaces. In each data marketplace, there are a few data sellers offering datasets for sale, and one data buyer interested in acquiring some of those datasets to train an ML model. The seller provides a pricing function that depends on the number of purchased samples. The buyer first decides how many data points to purchase from each seller given a data acquisition budget b. Then those data points are compiled into one dataset to train an ML model f(). The buyer also has a dataset Db to evaluate the performance of the trained model. Similar to real-world data marketplaces, the buyer can observe no sellers’ datasets but some summary information from the sellers. 

## 2. How to access the buyer's observation?

We provide a simple python library to access the buyer’s observation in each data marketplace. TO use it, (i) clone this repo, (ii) download the [data](https://drive.google.com/drive/folders/1DcML_lGqiwvN-l0KHE-WQG84gweydtZn?usp=sharing), (iii) unzip it, and (iv) place it under the folder ```marketinfo```.  Now, one can use the following code to specify the marketplace id

```
importDam
MyDam = Dam(instance=0)
```


The following code lists the buyer’s budget, dataset, and ml model.

```
budget = MyDam.getbudget()
buyer_data = MyDam.getbuyerdata()
mlmodel = MyDam.getmlmodel()
```


To list all sellers’ ids, execute 


```
sellers_id = MyDam.getsellerid()
```

To get seller i’s information, run

```
seller_i_price, seller_i_summary, seller_i_samples =  MyDam.getsellerinfo(seller_id=i)
```

seller_i_price contains the pricing function. seller_i_summary includes (i) the number of rows, (ii) the number of columns, (iii) the histogram of each dimension, and (iv) the correlation between each column and the label. Seller_i_samples contains 5 samples from each dataset.  

## 3. How to submit a solution?

The submission should contain K(=5) csv files. k.csv corresponds to the purchase strategy for the kth marketplace. Place all five submissions under your user name, and add a pull request to place them in \submission\. For example, one submission may look like


```

 \submission\lchen001\0.csv 

 \submission\lchen001\1.csv 

 \submission\lchen001\2.csv 

 \submission\lchen001\3.csv 

 \submission\lchen001\4.csv

```

Each csv file should contain one line of numbers, where the ith number indicates the number of data to purchase from the ith seller. For example, 0.csv containing

100,50,200,500
 
means buying 100, 50, 200, and 500 samples from seller 1, seller 2, seller 3, and seller 4 separately. 


## 4. How is a submission evaluated?

Once received the submission, we will first evaluate whether the strategy is legal (e.g., satisfying the budget constraint). Then we train the model on the dataset generated by the submitted strategy and evaluate its performance on the buyer’s data Db. We will report the performance averaged over all K marketplace instances. 

Requirements:

(i) you may use any (open-source/commercial) software;

(ii) you may not use external datasets;

(iii) do not create multiple accounts for submission;

(iv) follow the honor code (TODO: a clarification of the honor code);

(v) submit at most x times per week ;



## Contact and License
_DAM_ is Apache 2.0 licensed.

