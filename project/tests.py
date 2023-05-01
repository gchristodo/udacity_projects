"""
This is the tests written to check
the functionalitis of the churn_library.py
"""

import os
import logging
import time
from pathlib import Path
import pandas as pd
from churn_library import perform_eda, perform_feature_engineering


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(create_dataframe):
    '''
    test data import - this example is completed for you
    to assist with the other test functions
    '''
    try:
        dataframe = create_dataframe
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't have the proper shape")
        raise err


def test_eda(create_dataframe):
    '''
    test perform eda function
    '''
    dataframe = create_dataframe
    try:
        perform_eda(dataframe)
        time.sleep(5)
        logging.info("Testing perform_eda: SUCCESS")
    except Exception as err:
        logging.error("Testing perform_eda: FAILED")
        raise err
    folder_path = Path('./images/eda')
    try:
        assert any(folder_path.iterdir())
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: The directory doesn't contain any files.")
        raise err
    try:
        assert len(list(folder_path.iterdir())) == 6
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: The directory doesn't contain \n"
            "the correct num of images.")
        raise err


def test_encoder_helper(encode):
    '''
    test encoder helper
    '''
    try:
        data_encoded = encode
        logging.info("Testing encoder_helper: SUCCESS")
    except Exception as err:
        logging.error("Testing encoder_helper: FAILED")
        raise err
    try:
        assert isinstance(data_encoded, pd.DataFrame)
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The function must return a dataframe.")
        raise err
    try:
        assert data_encoded.empty is False
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The encoded dataframe is empty.")
        raise err


def test_perform_feature_engineering(encode):
    '''
    test perform_feature_engineering
    '''
    data_encoded = encode
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            data_encoded, 'Churn')
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except Exception as err:
        logging.error("Testing encoder_helper: FAILED")
        raise err
    try:
        assert isinstance(X_train, pd.DataFrame)
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: \n"
            "The X_train must be a dataframe.")
        raise err
    try:
        assert isinstance(X_test, pd.DataFrame)
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: \n"
            "The X_test must be a dataframe.")
        raise err
    try:
        assert isinstance(y_train, pd.Series)
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: \n"
            "The y_train must be a Series.")
        raise err
    try:
        assert isinstance(y_test, pd.Series)
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: \n"
            "The y_test must be a Series.")
        raise err
    try:
        assert X_train.empty is False
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: \n"
            "The X_train dataframe is empty.")
        raise err
    try:
        assert X_test.empty is False
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: \n"
            "The X_test dataframe is empty.")
        raise err
    try:
        assert y_train.empty is False
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: \n"
            "The y_train Series is empty.")
        raise err
    try:
        assert y_test.empty is False
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: \n"
            "The y_test Series is empty.")
        raise err


def test_train_models():
    '''
    test train_models
    '''
    folder_path = Path('./models')
    try:
        assert any(folder_path.iterdir())
        logging.info("Testing test_train_models: SUCCESS")
    except AssertionError as err:
        logging.error("Testing test_train_models: FAILED")
        raise err
    try:
        assert len(list(folder_path.iterdir())) == 3
        logging.info(
            "Testing test_train_models: The folder contains two models.")
    except AssertionError as err:
        logging.error(
            "Testing test_train_models: \n"
            "The directory doesn't contain the correct amount of pkl files.")
        raise err
    try:
        folder_path = './models'
        rfc_pkl = 'rfc_model.pkl'
        assert os.path.isfile(os.path.join(folder_path, rfc_pkl))
        logging.info(
            "Testing test_train_models: \n"
            "The folder contains the RFC model.")
    except AssertionError as err:
        logging.error(
            "Testing test_train_models: \n"
            "The folder doesn't contain the RFC model.")
        raise err
    try:
        folder_path = './models'
        lr_pkl = 'logistic_model.pkl'
        assert os.path.isfile(os.path.join(folder_path, lr_pkl))
        logging.info(
            "Testing test_train_models: \n"
            "The folder contains the LR model.")
    except AssertionError as err:
        logging.error(
            "Testing test_train_models: \n"
            "The folder doesn't contain the LR model.")
        raise err
