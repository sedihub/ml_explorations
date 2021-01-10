# Telco Customer Churn

For this notebook, you need to obtain the dataset. It can be acquired from either [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn) or [IBM](https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113).

This exploration is similar to the one on the [Boston Housing Prices](https://github.com/sedihub/ml_explorations/tree/main/boston_housing_prices) dataset, except this time we have a classification problem. We are interested in seeing how far we can get by using a simple DNN.

## Data Explorations Results

In order to be able to meaningfully assess the effectiveness of the DNN classifier on this task, we need to see how far we can get with conventional ML tools and in-depth feature analysis and engineering.

Feature analysis reveal some interesting, though not surprising, aspects listed below:

 - "TotalCharges" is a redundant feature:
!["Total Charges" is a redundant feature](https://github.com/sedihub/ml_explorations/blob/main/telco_customer_churn/.images/total_charges_is_redundant.png?raw=true)

 - The set of Internet-related features are fairly correlated. "StreamingTV" and "StreamingMovies" in particular are highly correlated:
!["Total Charges" is a redundant feature](https://github.com/sedihub/ml_explorations/blob/main/telco_customer_churn/.images/internet_related_features.png?raw=true)

 - Having contract inversely correlated with staying. The reverse (not having contract therefore canceling) is correlated but not as strongly. This is of course expected:
!["Total Charges" is a redundant feature](https://github.com/sedihub/ml_explorations/blob/main/telco_customer_churn/.images/contract_churn_scatter_plot.png?raw=true)
!["Total Charges" is a redundant feature](https://github.com/sedihub/ml_explorations/blob/main/telco_customer_churn/.images/contract_churn_confusion_matrix.png?raw=true)


## Upshot

We attempted to develop classifiers in two separate ways: Using random forest and gradient boosting after an in-depth exploration of the dataset. We also ignored the findings of feature explorations, and attempted developing a classifier using a DNN. Once can draw the following conclusions from the presented analysis here:

 - DNNs seem to be more prone to overfitting when dealing with classification problems with not much data. The reason that DNNs are more prone to overfitting is that they have a high capacity. 
 - DNNs have the important advantage of not being sensitive to feature engineering. They can learn the importance of various features provided that enough data is available. 
 - DNNs, not requiring much feature engineering, seem to offer an interesting route to fast solutions.
