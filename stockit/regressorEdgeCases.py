import unittest
from stockit_class import stockit_class
from dataGraber import returnData
from pandas import read_csv
import numpy as np

stockSymbols = ["NVDA","AMD","AMZN","ACB","ADBE"]

class testBrokenStuff(unittest.TestCase):

    # test if the output of the stockit regressor is a numpy array
    #def test_regressor(self):
    #    self.assertIsInstance(stockit_instance.predict(dataLength),float())
    # test training of linear regression models
    def testNotDefined(self):
        '''
        tests not found error that occurs when one attempts to make a prediction of the index that is the exact length of the DataFrame
        '''
        # for SSRR regressor
        for stock in stockSymbols:
            data = returnData(stock)
            s = stockit_class(data)
            s.train(SSRRbool=True)
            print(s.predict(len(data)))
        # for linear regressor
        for stock in stockSymbols:
            data = returnData(stock)
            s = stockit_class(data)
            s.train()
            print(s.predict(len(data)))
        # for SVR regressor
        for stock in stockSymbols:
            data = returnData(stock)
            s = stockit_class(data)
            s.train(SVRbool=True)
            print(s.predict(len(data)))
        '''
        tests regressing on an index 0-10 (11-1) times the max of the data
        '''
        print("FARDATES TESST__________")
        for i in range(11):
            # for SSRR regressor
            for stock in stockSymbols:
                data = returnData(stock)
                s = stockit_class(data)
                s.train(SSRRbool=True)
                print(s.predict(len(data) * i))
            # for linear regressor
            for stock in stockSymbols:
                data = returnData(stock)
                s = stockit_class(data)
                s.train()
                print(s.predict(len(data) * i))
            # for SVR regressor
            for stock in stockSymbols:
                data = returnData(stock)
                s = stockit_class(data)
                s.train(SVRbool=True)
                print(s.predict(len(data) * i))
        '''
        TEST FOR NEGATIVES
        '''
        print("NEGATIVE NUMBERS")
        for i in range(11,0,-1):
            for stock in stockSymbols:
                data = returnData(stock)
                s = stockit_class(data)
                s.train(SSRRbool=True)
                print(s.predict(i))
            # for linear regressor
            for stock in stockSymbols:
                data = returnData(stock)
                s = stockit_class(data)
                s.train()
                print(s.predict(len(data) * i))
            # for SVR regressor
            for stock in stockSymbols:
                data = returnData(stock)
                s = stockit_class(data)
                s.train(SVRbool=True)
                print(s.predict(len(data) * i))

unittest.main()
