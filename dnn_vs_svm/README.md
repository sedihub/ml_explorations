**TODO: Clean up the notebook (this is the super old notebook...)**
<br><br>

# Deep Neural Network vs. Support Vector Machine

Support vector machines (SVM) classifiers used in conjuncture with a suitable kernel (e.g., RBF) provide can be quite effective. Deep neural networks are also quite effective in capturing complicated dependencies. In this exploration, we compare the two on engineered and random datasets.

The three datasets used in this exploration are:
 - [Separable ellipsoidal point cloud shells](#separable-ellipsoidal-point-cloud-shells)
 - [Intersecting ellipsoidal point cloud shells](#intersecting-ellipsoidal-point-cloud-shells)
 - [Random point clouds](#random-point-clouds)


<br><br>
## Separable Ellipsoidal Point Cloud Shells

In this case we apply the SVM and DNN classifiers on two ellipsoidal point cloud distributions that are perfectly separable:

<p align="center">
    <img src="https://github.com/sedihub/ml_explorations/blob/main/dnn_vs_svm/.images/separable_pont_clouds.png" alt="Intersecting ellipsoidal point clouds" width="80%" height="80%">
</p>

After fitting the SVM classifier (left) to the data, and training the DNN (right) we obtain:
<p align="center">
</p>

For the DNN, we find:
<p align="center">
    <img src="https://github.com/sedihub/ml_explorations/blob/main/dnn_vs_svm/.images/svm_classification_results_on_separable_point_cloud.png" alt="SVM classification results on separable point clouds" width="49%" height="49%">
    <img src="https://github.com/sedihub/ml_explorations/blob/main/dnn_vs_svm/.images/dnn_classification_results_on_separable_point_cloud.png" alt="DNN classification results on separable point clouds" width="49%" height="49%">
</p>


<br><br>
## Intersecting Ellipsoidal Point Cloud Shells
<p align="center">
    <img src="https://github.com/sedihub/ml_explorations/blob/main/dnn_vs_svm/.images/intersecting_pont_clouds.png" alt="Separable ellipsoidal point clouds" width="80%" height="80%">
</p>

Trained SVM (left) and DNN (right):
<p align="center">
    <img src="https://github.com/sedihub/ml_explorations/blob/main/dnn_vs_svm/.images/svm_classification_results_on_intersecting_point_cloud.png" alt="SVM classification results on intersecting point clouds" width="49%" height="49%">
    <img src="https://github.com/sedihub/ml_explorations/blob/main/dnn_vs_svm/.images/dnn_classification_results_on_intersecting_point_cloud.png" alt="DNN classification results on intersecting point clouds" width="49%" height="49%">
</p>


<br><br>
## Random Point Clouds
<p align="center">
    <img src="https://github.com/sedihub/ml_explorations/blob/main/dnn_vs_svm/.images/random_pont_clouds.png" alt="Random point clouds" width="80%" height="80%">
</p>

The results for SVM (left) and DNN (right) classifiers:
<p align="center">
    <img src="https://github.com/sedihub/ml_explorations/blob/main/dnn_vs_svm/.images/svm_classification_results_on_random_point_cloud.png" alt="SVM classification results on random point clouds" width="49%" height="49%">
    <img src="https://github.com/sedihub/ml_explorations/blob/main/dnn_vs_svm/.images/dnn_classification_results_on_random_point_cloud.png" alt="DNN classification results on random point clouds" width="49%" height="49%">
</p>


