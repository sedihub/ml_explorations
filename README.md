# Machine Learning Explorations

This repository contains various small projects and explorations that I have done. These are broadly within machine learning and statistics. The main goal is to explore and approach problems often with well-established solutions from different and often unconventional angles. 

Here are some of the questions that motivated some of the projects in this repository:
 - The common wisdom is that one should rely on hand-crafted features and more generally conventional machine learning techniques when there is not enough data available to utilize deep neural networks. How do regression and classification tasks differ from this perspective? Can we gain some intuition into how much data is enough by trying DNNs on some of the benchmark datasets?
 - Gaussian processes are seem quite attractive. There is even an interesting [Distill](https://distill.pub/) publications of this topic: [A Visual Exploration of Gaussian Processes](https://distill.pub/2019/visual-exploration-gaussian-processes/). There are many Python libraries that provide GP, notably [SciKit Learn](https://scikit-learn.org/stable/modules/gaussian_process.html), [PyMC3](https://docs.pymc.io/) and now even in [TensorFlow](https://www.tensorflow.org/probability/examples/Gaussian_Process_Regression_In_TFP). The visualizations are surely stunning, but how do they work? How do these libraries come up with the confidence intervals?

Some of these date back to a few years ago. I have cleaned them up and made them available here so that I can share them with others more conveniently and also in case someone else finds them useful. There are still a number of these that I have to clean up and add.

Here is a list of these mini-projects:
 - [Fully-connected network for the *Boston Housing Prices* dataset.](https://github.com/sedihub/ml_explorations/tree/main/boston_housing_prices)
 - [DNN-based classifier for the *Telco Customer Churn* problem.](https://github.com/sedihub/ml_explorations/tree/main/telco_customer_churn)
 - [Gaussian process from scratch with only NumPy!](https://github.com/sedihub/ml_explorations/tree/main/gaussian_process_from_scratch)
 - [DNN vs. SVM](https://github.com/sedihub/ml_explorations/tree/main/dnn_vs_svm)
 - [Benchmarking Common Classifiers](https://github.com/sedihub/ml_explorations/tree/main/benchmarking_common_classifiers)
 - [Sci-Kit Learn Toy Datasets](https://github.com/sedihub/ml_explorations/tree/main/sklearn_toy_datasets)
 - [Learning metric in K-NN regression and classification problems](https://github.com/sedihub/ml_explorations/tree/main/learned_metric_in_knn) (coming soon!)
 - ...
