import unittest
from stockit_class import stockit_class 
from pandas import read_csv 
import numpy as np

data = read_csv("NVDA.csv")
dataLength = len(data)

stockit_instance = stockit_class(data)

class testStockit(unittest.TestCase):
    # test if the output of the stockit regressor is a numpy array 
    def test_regressor(self):
        self.assertIsInstance(stockit_instance.predict(dataLength),np.ndarray)
    def test_train(self):
        # testt that the .train() method will return 1
        self.assertEqual(stockit_instance.train(),1)
        # test that the .train() method with all parateters modified will still return 1  
        self.assertEqual(stockit_instance.train(5, 100, True),1)
