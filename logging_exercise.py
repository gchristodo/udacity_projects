"""
Provides some arithmetic functions
"""
import logging
import pandas as pd


logging.basicConfig(
    filename="./results.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s"
)


def read_data(file_path):
    """_summary_

    Args:
        file_path (_type_): _description_
    """
    try:
        dataframe = pd.read_csv(file_path)
        logging.info("SUCCESS: There are %s rows in your df.", dataframe.shape[1])
        logging.info("SUCCESS: Your file was successfully read!")
    except FileNotFoundError:
        logging.error("We were not able to find that file")
    return dataframe


if __name__ == '__main__':
    my_dataframe = read_data("settings_table_2.csv")
