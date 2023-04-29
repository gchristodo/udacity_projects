"""
Provides some arithmetic functions
"""
import logging


logging.basicConfig(
    filename="./results.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s"
)


def sum_vals(value_a, value_b):
    """_summary_

    Args:
        file_path (_type_): _description_
    """
    try:
        logging.info("%s %s", value_a, value_b)
        assert isinstance(value_a, int)
        assert isinstance(value_b, int)
        logging.info("SUCCESS: The values are int")
    except AssertionError:
        logging.info(value_a, value_b)
        logging.error("The values a and b are not integers")
    return value_a + value_b


if __name__ == '__main__':
    MY_VALUE = sum_vals(5, 4)
    print(MY_VALUE)
