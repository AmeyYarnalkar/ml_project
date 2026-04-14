from exceptions import CustomException
import sys
from logger import logging

try:
    print(1/0)

except Exception as e:
    error = CustomException(e, sys)
    logging.error(error)
    raise error