# Presence of Cardiovascular Disease
This project demonstrates hyperparameter optimization of Random Forest regression model using grid search and 5-fold cross validation. Scikit-Learn pipeline is used
to sequentially apply important feature transformation. The dataset was collected by the Hungarian Institute of Cardiology, University Hospital (Zurich, Switzerland), 
University Hospital (Zurich, Switzerland) and V.A. Medical Center (Long Beach and Cleveland Clinic Foundation). Principal investigators responsible for the data collection 
at each institution are:
1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation:Robert Detrano, M.D., Ph.D.

The dataset was obtained from UCI Machine Learning Repository, Center for Machine Learning and Intelligent Systems. Retrieved from https://archive.ics.uci.edu/ml/datasets/heart+disease
The target variable in the dataset represents presence of some cardiovascular disease. The features measured in the dataset are defined in the table:

| Variable | Definition                                                                           |
|----------|--------------------------------------------------------------------------------------|
| Age      | Age of patient                                                                       |
| Sex      | Gender                                                                               |
| cp       | Type of chest pain (typical angina, atypical angina, non-anginal pain, asymptomatic) |
| trestbps | Resting blood pressure                                                               |
| chol     | Serum cholesterol                                                                    |
| fbs      | Fasting blood sugar > 120mg/dl                                                       |
| restecg  | Resting Electrocardiography                                                          |
| thalach  | Max heart rate                                                                       |
| exang    | Exercise induced angina                                                              |
| oldpeak  | ST depression induced by exercise relative to rest                                   |
| slope    | Slope of peak exercise ST segment (unsloping, flat, downsloping)                     |
| ca       | Number of vessels colored by Fluoroscopy (0-3)                                       |
| thal     | Presence of fixed or reversible defect in stress echocardiography                    |  


## Blog 
My blog on this project can be accessed at https://mlai1.blogspot.com/2021/05/random-forest-classification-model-to.html


## Cross Validation
5-fold cross validation is used to avoid overfitting and to collect model evaluation metrics. In 5-fold cross validation, the training set is split
into 5 groups. In each iteration, one group is used as a hold-out set and the model is trained on the remaining groups. Evaluation metrics are collected
and the process is repeated. Overall performance of the model is evaluated based on the metrics.

## Hyperparameter Optimization
This Random Forest model includes hyperparameter optimization. In this optimization procedure, a grid search on a set of hyperparameters is performed in 
order to find model settings that achieve the best performance on a given dataset. It is important to note that 5-fold cross validation is used so that
the performance is evaluated on an independent hold-out test set. 

## Model Results and Feature Importance
Once the best model settings are picked, evaluation metrics are obtained for the training and testing set. This includes ROC-AUC, accuracy, recall and
precision. Additionally, numeric values of feature importances are also collected to highlight contribution of each feature to the model.

## Model Persistence
Model persistance can be performed using object serialization. A model is converted into byte stream and saved on the server. Shell commands can be used to 
encrypt the model file and to allow access to only specific users.
