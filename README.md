**CWT-XGBoost Dependencies:**  
•	Xgboost  
•	Pandas  
•	Numpy  
•	Sklearn  
•	Joblib  
•	Statsmodels

**Data Preparation:**  
•	The first column should contain the time data.  
•	The last column should contain the target parameter values.  
•	The middle columns should contain the feature values.

**Input File Paths:**  
•	Provide the file paths for the training data, testing data, output file, and machine learning model on lines 11-14, respectively.

**Training Process:**  
•	Adjust the input hyperparameter values in lines 29-43 as needed.  
•	Call the XGBoost_training() function to execute the training process.

**Prediction Process:**  
•	Call the XGBoost_predicting() function to generate predictions.

**Output Data:**  
•	The first column will contain the time data.  
•	The second column will contain the raw target parameter values.  
•	The last column will contain the machine learning model results.
