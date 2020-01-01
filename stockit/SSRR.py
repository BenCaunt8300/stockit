import numpy as np
import pandas as pd
from statistics import mean
import random
from tqdm import tqdm
from math import fabs
import matplotlib.pyplot as plt
class SSRR:
    '''serialized slope randomized regression implementation developed by Ben Caunt.

    The SSRR algorithm works as such, for each given data point, calculate the slope between each point then average them together

    Set b equal to the last data point in the set.
    When making a prdiction,  for each index after the final index in the dataset,
    add a random bias (+ or - the Mean Absolute Deviation) to each indicie up to the final predicted indicie and include the slopes between these predicted prices as part of the slope average calculated with the data
    use y = mx + b to calculate result
    Invented by Ben Caunt  '''

    def __init__(self):
        # m of y = mx+b
        self.slopeAvg = None
        # bb of y = mx+b
        self.b = None

        self.data = None
    def fit(self, data):
        '''
        fit model to data

        data is the data passed into the regressor
        index is the number of indicies after the start that the regressor should start training on
        '''
        self.data = np.array(data)
        # numpy array containing the slopes between each of the given points

        slopeArray = np.zeros(len(self.data)-1)
        for count, price in tqdm(enumerate(self.data)):
            try:
                # slope between current and next point
                currentToNextSlope = calcSlope(count, price, count + 1, self.data[count + 1])
                # change current value of slope array to the current slope instead of the default 0
                slopeArray[count] = currentToNextSlope
            except IndexError:
                print("index error found, potentially dangerous but maybe not idk")
        # the y intercept for this example is equal to the last indicie of the data
        self.slopeAvg = mean(slopeArray)
        self.b = yintercept(self.slopeAvg,self.data[-1],len(self.data))

    def predict(self, x):
        '''
        uses model to make prediction about future prices using SSRR estimator
        '''
        predicted = []
        # if the predicted incidie is part of the training data, simply sample that
        if len(self.data) > x:
            return self.data[x]
        # use the fancy pants regressor i developed
        else:
            # amount that the slope is devided by the keep it normalized and prevent excessive things from happening
            slopeScale = 5
            self.slopeAvg /= slopeScale
            # prediction for the first indicie using y = mx + b
            prediction = self.slopeAvg*(len(self.data)+1)+self.b
            # increment through the indicies from the max of the data to the predicted point in question
            for i in range(len(self.data),x,3):
                # prediction with added variance / random change
                variedPrediction = self.addMAD(prediction)
                # calculate slope between the previous indicie prediction and the new varied prediction
                variedVsPreviousSlope = calcSlope(i, prediction, i + 1, variedPrediction)
                # scale variedVsPreviousSlope
                variedVsPreviousSlope /= slopeScale
                # average in new slope with training slope average
                self.slopeAvg = (self.slopeAvg + variedVsPreviousSlope) / 2
                print(f"slopeAvg {self.slopeAvg}")
                print(f"i {i}")
                print(f"b {self.b}")
                # re-calculate prediction
                prediction = fabs((self.slopeAvg*i)+self.b) / slopeScale
                predicted.append(prediction)
            # final prediction with scaled slope
            prediction = fabs((self.slopeAvg*i)+self.b) / slopeScale

            return prediction
    def addMAD(self, dataPoint):
        '''
        adds / subtracts 0 to the MAD to the point
        '''
        # change to float
        dataPoint = float(dataPoint)
        # amount that the data actually changes by
        varianceAdded = random.choice([random.random() * numpyMad(self.data),random.random() * -numpyMad(self.data)])
        return dataPoint + varianceAdded

def numpyMad(data):
    return np.mean(np.absolute(data - np.mean(data)))



def calcSlope(x1,y1,x2,y2):
    y = y2-y1
    x = x2-x1
    slope = y/x
    return slope
# adds random values to datapoints for algorithm to have randomness

    # final result of change + dataPoint
    return dataPoint + change
# finds and returns y intercept
def yintercept(m,y1,x1):
    '''
    calculate y intercept with:
    slope
    y1
    x2
    '''
    b=y1-m*x1
    return b

def main():
    data = pd.read_csv("NVDA.csv")
    data = data.close
    regressor = SSRR()
    regressor.fit(data)
    # predict 50 days out 
    print(regressor.predict(len(data) + 50))

if __name__ == '__main__':
    main()
