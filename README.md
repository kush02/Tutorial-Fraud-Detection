# Tutorial-Fraud-Detection

# Introduction
In this tutorial, we will learn about how to build a classifier for fraud detection. Fraud detection is important so that people don’t lose money.

I will be using the Credit Card Fraud Detection dataset from Kaggle. The data was collected and analyzed during a research collaboration between Worldline and the Machine Learning Group of ULB on big data mining and fraud detection.

The tutorial will cover the following tasks:
  1.	Bird’s eye view of the dataset
  2.	Exploratory data analysis (EDA)
  3.	Modelling
  4.	Evaluation of different ML algorithms
  5.	Next steps
  
# A Brief Look At The Dataset:
The dataset is stored in a csv format. It contains 31 columns. There is 1 target variable and 30 potential predictors (potential because we may remove features that have no correlation with the target variable). The creator of the dataset could not provide the original features due to confidentiality issues. Thus, the creator has provided us with PCA-transformed features. Hence all features are numerical.

Now it’s time for EDA!

# Exploring The Data:
There are 31 columns in the dataset. The names of the columns are
 
The PCA-transformed features are *V1-V28*. Features like Time, Amount and Class have not been transformed.

