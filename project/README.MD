# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This is the refactored version of churn_notebook. It is written
and organized in a way to be readable and maintainable. There are
distinct section for each functionality so that the user can easily
refactor and track bugs.

## Files and data description
### data
This folder contains the input dataset: bank_data.csv
### images
This folder contains two folders:
 - eda: This folder contains images from the EDA process
 - results: This folder contains images regarding Classification reports, ROC curves and Feature Importances for two models
### logs
This folder contains the churn_library.log file which documents whether the execution of churn_script_logging_and_tests.py was successful.
### models
This folder contains the trained and saved Random Forest and Logistic Regression models.
### churn_library.py
This file contains all the helper functions which are used to build the main.py
### churn_script_logging_and_tests.py
This file contains all the necessary tests to check the integrity of the flow and the unit functionality.
### conftest.py
This file contains all the fixtures used by the churn_script_logging_and_tests.py. 
### constants.py
This file contains a dictionary which includes all the global variables of the project.
### main.py
This file acts as a wrapper. 
### README.md
The current file.
### requirements_py3.6.txt
The requirements to be installed if using python 3.6.
### requirements_py3.8.txt
The requirements to be installed if using python 3.8.


## Running Files
 - Install requirements: `python -m pip install -r requirements_py3.6.txt`
 - Run entire flow: `python main.py`
 This is the main function which imports the dataset, performs EDA and feature engineering,
 trains the models and produces classification reports, feature importances and ROC-AUC score as images.
 If you want to skip the training process and just load the already trained models, comment line 25 in 
 main.py file.
 - Run test: `pytest churn_script_logging_and_tests.py -v`
 By running this command, you will get an analytic report with the tests conducted
 and whether they have passed or failed. The log file in logs folder contains the specifics for test failures and successes.



