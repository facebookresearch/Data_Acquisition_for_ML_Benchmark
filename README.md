# dataperf-dam: A Data-centric Benchmark on Data Acquisition for Machine Learning

This github repo serves as the starting point for submissions and evaluations for data acquisition for machine learning benchmark, or in short, DAM, as part of the DataPerf benchmark suite [https://dataperf.org/](https://dataperf.org/)


## 1. What is the DAM benchmark?

An increasingly large amount of data is purchased for AI-enabled data science applications. How to select the right set of datasets for AI tasks of interest is an important decision that has, however, received limited attention. A naive approach is to acquire all available datasets and then select which ones to use empirically. This requires expensive human supervision and incurs prohibitively high costs, posing unique challenges to budget-limited users. 

How can one decide which datasets to acquire before actually purchasing the data to optimize the performance quality of an ML model?  In the DAM (Data-Acquisition-for-Machine-learning) benchmark, the participants are asked to tackle the aforementioned problem. Participants need to provide a data purchase strategy for a data buyer in K (=5 in the beta version) separate data marketplaces. In each data marketplace, there are a few data sellers offering datasets for sale, and one data buyer interested in acquiring some of those datasets to train an ML model. The seller provides a pricing function that depends on the number of purchased samples. The buyer first decides how many data points to purchase from each seller given a data acquisition budget b. Then those data points are compiled into one dataset to train an ML model f(). The buyer also has a dataset Db to evaluate the performance of the trained model. Similar to real-world data marketplaces, the buyer can observe no sellers’ datasets but some summary information from the sellers.

## 2. How to participate this challenge?
We suggest to start participating by using the [colab notebook](https://colab.research.google.com/drive/1HYoFfKwd9Pr-Zg_e2uJxWF8yHqa9sRMn?usp=sharing). It is self-contained, and shows how to (i) install the needed library, (ii) access the buyer's observation, and (iii) create strategies ready to be submitted. In the following we explain this in more details. 

## 3. How to access the buyer's observation?

We provide a simple python library to access the buyer’s observation in each data marketplace.
To use it, we recommand to create a virtual environment by
```
conda create -n DAM python=3.8
conda activate DAM
```
and then clone the github repo and install all libraries, and download the data by
```
git clone https://github.com/facebookresearch/Data_Acquisition_for_ML_Benchmark
cd Data_Acquisition_for_ML_Benchmark
pip install -r requirements.txt
wget https://github.com/lchen001/Data_Acquisition_for_ML_Benchmark/releases/download/v0.0.1/marketinfo.zip
! unzip marketinfo.zip
cd src
```

Now, one is ready to use this library. For example,  to specify the marketplace id, one can use

```
from dam import Dam
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

Note: For simplification purposes, all sellers sell the same type of data, or in a more mathematically way, their data distribution shares the same support. For example, the number of columns are the same, and so the semantic meaning.

More details on the price function: given a sample size, the price can be calculated by calling the get_price_samplesize function. For example, if the sample size is 100, then calling

```
seller_i_price.get_price_samplesize(samplesize=100)
```
gives the price.

More details on the seller summary: the seller_i_summary contains four fields as follows:

```
seller_i_summary.keys()
>>> dict_keys(['row_number', 'column_number', 'hist', 'label_correlation'])
```
Here, seller_i_summary['row_number'] encode the number of data points. Similarly, seller_i_summary['column_number'] equals the number of features plus (the label). seller_i_summary['hist'] is a dictionary containg the histgram for each feature. seller_i_summary['label_correlation'] is a dictionary that represents the pearson correlation between each feature and the label.

For example, one can print the histogram of the second feature by 
```
print(seller_i_summary['hist']['2'])
>>> {'0_size': 3, '1_size': 35, '2_size': 198, '3_size': 821, '4_size': 2988, '5_size': 8496, '6_size': 11563, '7_size': 5155, '8_size': 704, '9_size': 37, '0_range': -0.7187578082084656, '1_range': -0.5989721298217774, '2_range': -0.4791864514350891, '3_range': -0.3594007730484009, '4_range': -0.23961509466171266, '5_range': -0.11982941627502441, '6_range': -4.373788833622605e-05, '7_range': 0.11974194049835207, '8_range': 0.23952761888504026, '9_range': 0.35931329727172856, '10_range': 0.47909897565841675}
```
How to read this? This representation basically documents (i) how the histogram bins are created (i_range), and (ii) how many points fall into each bin (i_size). For example, '2_size':198 means 198 data points are in the 2nd bin, and '' '2_range': -0.4791864514350891, '3_range': -0.3594007730484009'' means the 2nd bin is within [-0.4791864514350891,-0.3594007730484009].

```
print(seller_i_summary['label_correlation']['2'])
>>> 0.08490820825406746
```
This means the correlation between the 2nd feature and the label is 0.08490820825406746.

Note that all features in the sellers and buyers' datasets are NOT in their raw form. In fact, we have extracted those features using a deep learning model (more specifically, a dist-bert model) from their original format.

## 3. How to submit a solution?

The submission should contain K(=5) txt files. k.txt corresponds to the purchase strategy for the kth marketplace. The notebook will automatically generate txt files for submission under the folder ```\submission\my_submission```. For example, one submission may look like


```

 \submission\my_submission\0.txt 

 \submission\my_submission\1.txt 

 \submission\my_submission\2.txt 

 \submission\my_submission\3.txt 

 \submission\my_submission\4.txt

```

Each txt file should contain one line of numbers, where the ith number indicates the number of data to purchase from the ith seller. For example, 0.txt containing

```
100,50,200,500
```

means buying 100, 50, 200, and 500 samples from seller 1, seller 2, seller 3, and seller 4 separately. 

Once you are ready, upload the txt files to DynaBench for evaluation: https://dynabench.org/tasks/DAM/


## 4. How is a submission evaluated?

Once received the submission, we will first evaluate whether the strategy is legal (e.g., satisfying the budget constraint). Then we train an ML model on the dataset generated by the submitted strategy and evaluate its performance (standard accuracy) on the buyer’s data Db. We will report the performance averaged over all K marketplace instances. 

What ML model to train? To focus on the data acquisition task, we train a simple logistic regression model. More specifically, we use the following model 

```
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0)	
```  

Requirements:

(i) you may use any (open-source/commercial) software;

(ii) you may not use external datasets;

(iii) do not create multiple accounts for submission;

(iv) follow the honor code.

## Contact and License
_DAM_ is Apache 2.0 licensed.

