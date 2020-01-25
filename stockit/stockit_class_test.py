import unittest
from stockit_class import stockit_class
from pandas import read_csv
import numpy as np

data = read_csv("TSLA.csv")
dataLength = len(data)


class testStockit(unittest.TestCase):

    # test if the output of the stockit regressor is a numpy array
    #def test_regressor(self):
    #    self.assertIsInstance(stockit_instance.predict(dataLength),float())
    # test training of linear regression models
    def test_linregTrain(self):
        stockit_instance = stockit_class(data)

        # testt that the .train() method will return 1
        self.assertEqual(stockit_instance.train(),1)

        self.assertEqual(stockit_instance.train(SVRbool=False,SSRRbool=False),1)
        # make sure predicting yields no errors
        print(stockit_instance.predict(dataLength + 10))

        print(stockit_instance.predict(dataLength - 10))
    #test training of svr model
    def test_svrTrain(self):
        stockit_instance = stockit_class(data)
        self.assertEqual(stockit_instance.train(SVRbool = True),1)
        # make sure predicting yields no errors
        print(stockit_instance.predict(dataLength + 10))

        print(stockit_instance.predict(dataLength - 10))
    # test training of SSRR custom model
    def test_SSRRTrain(self):
        stockit_instance = stockit_class(data)
        print(data.Close.tail())
        self.assertEqual(stockit_instance.train(index=100,SSRRbool=True),1)
        self.assertEqual(stockit_instance.train(index=0,SSRRbool=True),1)
        # test if predicting on an indicie less than the max results in output equal to known indicie
        #self.assertEqual(stockit_instance.predict(dataLength - 10), data['Close'][len(data)-10])
        # make sure predicting yields no errors
        print(stockit_instance.predict(dataLength + 10))
