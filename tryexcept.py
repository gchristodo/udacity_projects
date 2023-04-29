import pandas as pd
import logging

logging.basicConfig(
    filename='./results.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)

def read_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print(df.head())
        return df
    except FileNotFoundError:
        print("We were not able to find that file")