# library doc string
"""
This is the submission of George Christodoulou.
"""

# import libraries
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from constants import constants
sns.set()
os.environ["QT_QPA_PLATFORM"] = "offscreen"


def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    data = pd.read_csv(pth)
    return data


def perform_eda(data):
    """
    perform eda on df and save figures to images folder
    input:
            data: pandas dataframe

    output:
            None
    """
    data["Churn"] = data["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    plt.figure(figsize=(20, 10))
    data["Churn"].hist()
    plt.savefig("images/eda/Churn_hist.png")
    plt.figure(figsize=(20, 10))
    data["Customer_Age"].hist()
    plt.savefig("images/eda/Customer_Age_hist.png")
    plt.figure(figsize=(20, 10))
    data.Marital_Status.value_counts("normalize").plot(kind="bar")
    plt.savefig("images/eda/Marital_Status_hist.png")
    plt.figure(figsize=(20, 10))
    sns.histplot(data["Total_Trans_Ct"], stat="density", kde=True)
    plt.savefig("images/eda/Total_Trans_Ct_hist.png")
    plt.figure(figsize=(20, 10))
    sns.heatmap(data.corr(), annot=False, cmap="Dark2_r", linewidths=2)
    plt.savefig("images/eda/Correlation_Matrix.png")


def encoder_helper(data, category_lst, response):
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the
    notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be
            used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    """
    for category in category_lst:
        category_lst = []
        category_groups = data.groupby(category).mean()[response]

        for val in data[category]:
            category_lst.append(category_groups.loc[val])

        data[f"{category}_{response}"] = category_lst
    return data


def perform_feature_engineering(data, response):
    """
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could
                        be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    taget_variable = data[response]
    X_features = pd.DataFrame()
    keep_cols = constants["keep_cols"]

    X_features[keep_cols] = data[keep_cols]
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, taget_variable, test_size=0.3, random_state=42
    )
    return X_train, X_test, y_train, y_test


def classification_report_helper(report, output_pth):
    """
    helper function to turn each report to an image

    input:
            report: string
            output_pth: the save path

    output:
            None
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    table_data = [row.split() for row in report.split("\n")[2:-1]]
    table_data[2] = [
        "",
        "",
        "",
        "",
        "",
    ]
    table_data[3].insert(1, "")
    table_data[3].insert(2, "")
    table_data[4] = [" ".join(table_data[4][:2])] + table_data[4][2:]
    table_data[5] = [" ".join(table_data[5][:2])] + table_data[5][2:]
    ax.table(
        cellText=table_data,
        colLabels=["", "precision", "recall", "f1-score", "support"],
        bbox=[0, 0, 1, 1],
    )
    plt.savefig(output_pth)


def classification_report_image(
    y_train,
    y_test,
    y_train_preds_lr,
    y_train_preds_rf,
    y_test_preds_lr,
    y_test_preds_rf,
):
    """
    produces classification report for training and testing results
    and stores report as image in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    # Random Forest results
    # Test results
    report = classification_report(y_test, y_test_preds_rf)
    classification_report_helper(report, "images/results/RF_Test.png")
    # Train results
    report = classification_report(y_train, y_train_preds_rf)
    classification_report_helper(report, "images/results/RF_Train.png")
    # Logistic Regression
    # Test results
    report = classification_report(y_test, y_test_preds_lr)
    classification_report_helper(report, "images/results/LR_Test.png")
    # Train results
    report = classification_report(y_train, y_train_preds_lr)
    classification_report_helper(report, "images/results/LR_Train.png")


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    if model.__class__.__name__ == RandomForestClassifier().__class__.__name__:
        # Calculate feature importances if model is RF
        importances = model.feature_importances_
    elif model.__class__.__name__ == LogisticRegression().__class__.__name__:
        # Calculate feature importances -> LR
        importances = model.coef_[0]
    else:
        raise ValueError("Only random_forest and \n"
                         "logistic_regression are supported.")
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]
    # Create plot
    plt.figure(figsize=(20, 5))
    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)
    rfc_model = cv_rfc.best_estimator_
    lr_model = lrc
    joblib.dump(rfc_model, './models/new/rfc_model.pkl')
    joblib.dump(lrc, './models/new/logistic_model.pkl')
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(rfc_model, X_test, y_test, ax=ax, alpha=0.8)
    plot_roc_curve(lr_model, X_test, y_test, ax=ax, alpha=0.8)
    plt.savefig("images/results/ROC_AUC.png")
