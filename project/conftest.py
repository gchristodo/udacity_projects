"""
This file contains the fixtures used
in the churn_script_logging_and_tests.py 
"""

from churn_library import import_data, encoder_helper
from pytest import fixture
from constants import constants


@fixture(scope="session")
def create_dataframe():
    """This fixture creates a dataframe
    by importing the corresponding csv file.

    Returns:
        pd.Dataframe: The output dataframe
    """
    dataframe = import_data(constants['path'])
    return dataframe


@fixture(scope="session")
def encode(create_dataframe):
    """This fixture gets as input the above
    fixture and returns the encoded dataframe.

    Args:
        create_dataframe (function): A fixture used
        to create a dataframe

    Returns:
        pd.Dataframe: The encoded dataframe
    """
    dataframe = create_dataframe
    category_lst = constants['cat_columns']
    data_encoded = encoder_helper(dataframe, category_lst, 'Churn')
    return data_encoded
