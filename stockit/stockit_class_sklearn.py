import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import svm


#defines our regressor class
class stockit_class():
    #regressor class init function
    def __init__(self, data):
        self.data = data
    #returns mean of the dataset
    def mean(self):
        data = self.data
        length = len(data)
        total = sum(data)
        return total/length
    def MAD(x):
        #mean function for the mad function, cannot be used outside of the MAD function
        def mean_mad(z):
            #sum of the dataset
            total = sum(z)
            #length of the dataset
            length = len(z)
            #mean of the dataset
            return total/length

        #stores mean of x as a variable
        average = mean_mad(x)
        #creates an empty list that will hold each standard deviation
        devi_lst = []
        #increments through x and finds the distance between each index and the mean and appends them to 'devi_lst'
        for i in range(len(x)):
            print('deviation of {0} is {1}'.format(x[i],math.sqrt((average-x[i])**2)))
            devi_lst.append((math.sqrt((average-x[i])**2)))

        #the final mean absolute deviation
        return mean_mad(devi_lst)

    #training function
    def train(self, degree, index = 0):
        data = self.data

        #if index is equal to 0 then do things as normally
        if index == 0:
            x = []

            y = data
            #creates the x or independed variable
            for i in (range(0,len(data))):
                x.append(i)

            x = np.array(x)
            y = np.array(y)

            #reshape data
            x = x.reshape(-1,1)
            y = y.reshape(-1,1)
            '''
            if index is not equal to 0 then starting from the end of the dataset,
            increment back for the range of the index variable
            '''
        else:
            x_lst = []
            y_lst = []

            y = data

            for i in (range(index)):
                #the maximum index is equal to the data length
                max = len(y)

                distance_back = index-i
                x_lst.append(max-distance_back)
            y = pd.DataFrame(y)
            y_lst = y.tail(index)

            global x_index
            global y_index

            x_index = np.array(x_lst)
            y_index = np.array(y_lst)

            #reshape data
            x_index = x_index.reshape(-1,1)
            y_index = y_index.reshape(-1,1)

            x = x_index
            y = y_index


        lin = LinearRegression()
        lin.fit(x,y)
        global poly
        poly = PolynomialFeatures(degree = degree)
        global x_poly
        x_poly = poly.fit_transform(x)
        poly.fit(x_poly, y)
        global lin2
        lin2 = LinearRegression()
        lin2.fit(x_poly, y)

    def predict(self, predictor):
        pred = predictor
        pred = np.array(pred)
        pred = pred.reshape(1,-1)
        pred_poly = poly.fit_transform(pred)
        print(pred_poly.shape)

        output = lin2.predict(pred_poly)
        return output


def main():

    #creates pandas dataframe
    stock = 'VSR.V.csv'
    df = pd.read_csv(stock)
    #the last index of a dataset is equal to its length - ya bois law
    max = len(df)
    #prints the length of the dataset
    print("df length is: {0}".format(len(df)))

    stockit = stockit_class(df)

    stockit.train(degree = 10, index=300)
	#asks the model to train up to 3000 and make a prediction on 4000
    point_in_question = max+1
    point_prediction = stockit.predict(point_in_question)
    print(point_prediction)

    #creates the x or independed variable

    '''
    for i in tqdm(range(0,len(df))):
        x.append(i)
    x = np.array(x)
    y = np.array(y)
    #reshape data
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    '''

    predictions = lin2.predict(np.sort(x_poly, axis = 0))
    plt.title(stock)
    plt.plot(x_index, predictions, label = "predictions")
    plt.plot(x_index, y_index, label= "real")
    plt.scatter([point_in_question], [point_prediction], label = 'stockit.predict[{0}]'.format(point_in_question))
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
