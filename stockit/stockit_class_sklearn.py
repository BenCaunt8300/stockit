import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from matplotlib.pyplot import style
from statistics import mean
from sklearn import svm
import math


#creates stockit class
class stockit_class():
    #regressor class init function
    def __init__(self, data):
        #exception handler for finding the close column of a pandas dataframe
        try:
            data = data.close
        except:
            try:
                data = data.Close
            except:
                pass

        self.data = data
        self.poly = True

    #returns mean of the dataset
    def mean(self):
        data = self.data
        length = len(data)
        total = sum(data)
        return total/length

    def MAD(self):
        #mean function for the mad function, cannot be used outside of the MAD function
        def mean_mad(z):
            #sum of the dataset
            total = sum(z)
            #length of the dataset
            length = len(z)
            #mean of the dataset
            return total/length
        x = self.data
        #stores mean of x as a variable
        average = mean_mad(x)
        #creates an empty list that will hold each deviation
        devi_lst = []
        #increments through x and finds the distance between each index and the mean and appends them to 'devi_lst'
        for i in range(len(x)):
            print('deviation of {0} is {1}'.format(x[i],math.sqrt((average-x[i])**2)))
            devi_lst.append((math.sqrt((average-x[i])**2)))

        #the final mean absolute deviation
        return mean_mad(devi_lst)

    #poly regressor training function
    def train_poly(self, degree = 10, index = 0):
        self.poly = True
        data = self.data

        #if index is equal to 0 then do things as normally
        if index == 0:
            x = []

            y = data
            #creates the x or independed variable
            for i in tqdm(range(len(data))):
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
            max = len(y)
            for i in tqdm(range(index)):
                #the maximum index is equal to the data length

                distance_back = index-i
                x_lst.append(max - distance_back)
            try:
                y_lst = y.tail(index)
            except:
                y_lst = pd.DataFrame(y)
                y_lst = y_lst.tail(index)

            global x_index
            global y_index

            x_index = np.array(x_lst)
            y_index = np.array(y_lst)

            #reshape data
            x_index = x_index.reshape(-1,1)
            y_index = y_index.reshape(-1,1)

            x = x_index
            y = y_index


        global poly
        poly = PolynomialFeatures(degree = degree)
        global x_poly
        x_poly = poly.fit_transform(x)
        poly.fit(x_poly, y)
        global reg
        reg = LinearRegression()
        reg.fit(x_poly, y)
    #linear regressor training function
    def train_linear(self, index = 0):
        #set poly = to false so that the predict method knows not to use the special way of accomplishing the same thing but for polynomials
        self.poly = False
        data = self.data

        #if index is equal to 0 then do things as normally
        if index == 0:
            x = []

            y = data
            #creates the x or independed variable
            for i in tqdm(range(len(data))):
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

        #if the index is != to 0 then create a new dataset with the last 'index' items

        else:
            x_lst = []
            y_lst = []

            y = data
            max = len(y)
            for i in tqdm(range(index)):
                #the maximum index is equal to the data length

                distance_back = index-i
                x_lst.append(max - distance_back)
            try:
                y_lst = y.tail(index)
            except:
                y_lst = pd.DataFrame(y)
                y_lst = y_lst.tail(index)

            global x_index
            global y_index

            x_index = np.array(x_lst)
            y_index = np.array(y_lst)

            #reshape data to work with sklearn
            x_index = x_index.reshape(-1,1)
            y_index = y_index.reshape(-1,1)

            x = x_index
            y = y_index

        global reg

        reg = LinearRegression()
        #fit regressor
        reg.fit(x, y)

    def predict(self, predictor):
        pred = predictor
        pred = np.array(pred)
        pred = pred.reshape(1,-1)
        if self.poly:
            pred_poly = poly.fit_transform(pred)
            pred  = pred_poly
        else:
            pass

        print("prediction made lol")

        output = reg.predict(pred)
        return output
    #uses moving average in combination with polynomial regression
    #creates dataset of the moving average values then applies polynomial regression
    def MA_reg(self, reg_degree = 10, reg_index = 0, MA_index = 100):
        data = self.data

        '''
        MOVING AVERAGE
        basically same as the real moving average method
        '''
        #for graphing the real price i think lol
        x_data_graphing = []
        for data_len in range(len(data)):
            x_data_graphing.append(data_len)


        #calculate moving average for duration of the thing

        #list of all the moving average values
        moving_avg_values = []
        #fill the first 'range(index)' with 0s because why not lol
        for fillerboi in range(MA_index):
            moving_avg_values.append(0)

        '''
        here is where we calculate the moving average for every 'window' of the dataset
        basically we start with the counter variable 'z' + index to get the starting position
        then we go back and average the past 20 positions from the starting variable and then save it to the list
        'moving_avg_values'
        '''

        for z in range(len(data)):
            #start 20 after the start of the datset
            current_pos = z+MA_index
            #holds the values of every 20 data points
            try:
                index_values = []
                for y in range(0,MA_index):
                    print(f"current_pos-x == {current_pos-y}")
                    index_values.append(data[current_pos-y])
                print(f"mean(index_values) == {mean(index_values)} ")
                moving_avg_values.append(mean(index_values))
            except:
                #dont worry about this
                print("stuff happens, moving on")
                #get out of here lol
                #we've gone as far as we can, stop here, youre wasting CPU time

        self.data = moving_avg_values
        '''
        polynomial regression
        '''

        #train the poly regressor on the moving average values
        self.train_poly(degree = reg_degree, index = reg_index)

    #moving average, input the number of time stamps with the 'index' variable
    def moving_avg(self, index = 100, show_plt = True, save_plt = False, name = "name", save_index = 90, save_dpi = 800):

        '''
        /**
        * For the brave souls who get this far: You are the chosen ones,
        * the valiant knights of programming who toil away, without rest,
        * fixing our most awful code. To you, true saviors, kings of men,
        * I say this: never gonna give you up, never gonna let you down,
        * never gonna run around and desert you. Never gonna make you cry,
        * never gonna say goodbye. Never gonna tell a lie and hurt you.
        */
        '''

        style.use("ggplot")

        #always document your code kids
        #oh yea, this is some moving average thing lol
        #it goes back x days, finds the average, graphs it
        #pretty lame but simpler than a neural network
        # **laughs in shape errors**
        '''
        basically this function takes in the input of a list and finds the average of it,
        the only difference from the standard mean function is it has the optimization of not having to calculate the length of the datset each time
        the length of the data that is being average is decided by the index variable
        '''

        data = self.data

        #for graphing the real price i think lol
        x_data_graphing = []
        for data_len in range(len(data)):
            x_data_graphing.append(data_len)

        x = []

        #calculate moving average for duration of the thing

        #list of all the moving average values
        moving_avg_values = []
        #fill the first 'range(index)' with 0s because why not lol
        for fillerboi in range(index):
            moving_avg_values.append(0)

        '''
        here is where we calculate the moving average for every 'window' of the dataset
        basically we start with the counter variable 'z' + index to get the starting position
        then we go back and average the past 20 positions from the starting variable and then save it to the list
        'moving_avg_values'
        '''

        for z in range(len(data)):
            #start 20 after the start of the datset
            current_pos = z+index
            #holds the values of every 20 data points
            try:
                index_values = []
                for y in range(index):
                    print(f"current_pos-x == {current_pos-y}")
                    index_values.append(data[current_pos-y])
                print(f"mean(index_values) == {mean(index_values)} ")
                moving_avg_values.append(mean(index_values))
            except:
                #dont worry about this
                print("stuff happens, moving on")
                #get out of here lol
                #we've gone as far as we can, stop here, youre wasting CPU time


        #fill in the x values for graphing
        for length_mov_avg_val in range(len(moving_avg_values)):
            x.append(length_mov_avg_val)

        #debug stuff, uncomment if you need lol

        print(f"len(x) = {len(x)}")
        print(f"len(moving_avg_values) = {len(moving_avg_values)}")

        if save_plt:
            x = pd.DataFrame(x)
            moving_avg_values = pd.DataFrame(moving_avg_values)
            x_data_graphing = pd.DataFrame(x_data_graphing)
            x = x.tail(save_index)
            data = data.tail(save_index)
            moving_avg_values = moving_avg_values.tail(save_index)
            x_data_graphing = x_data_graphing.tail(save_index)

        plt.plot(x, moving_avg_values, label = f"moving average {index}")
        plt.plot(x_data_graphing, data, label = "real values")

        if show_plt:
            plt.legend()
            plt.show()
        if save_plt:
            plt.legend()
            plt.savefig(name, dpi = save_dpi)

def main():

    #creates pandas dataframe
    stock = 'NVDA'

    df = pd.read_csv(f"{stock}.csv")
    #the last index of a dataset is equal to its length - ya bois law
    max = len(df)
    #prints the length of the dataset
    print("df length is: {0}".format(len(df)))

    stockit = stockit_class(df)

    def poly_regressor_demo():
        style.use('ggplot')
        stockit.train_poly(degree = 10, index=100)
    	#asks the model to train up to 3000 and make a prediction on 4000
        point_in_question = max+1
        point_prediction = stockit.predict(point_in_question)
        print(point_prediction)
        predictions = reg.predict(np.sort(x_poly, axis = 0))
        plt.title(stock)
        plt.plot(x_index, predictions, label = "poly reg predictions")
        plt.plot(x_index, y_index, label= "real")
        plt.scatter([point_in_question], [point_prediction], label = 'stockit.predict[{0}]'.format(point_in_question))
        plt.legend()
        plt.show()
    def linear_regressor_demo():
        style.use('ggplot')
        stockit.train_linear(index=100)
        point_in_question = max+1
        point_prediction = stockit.predict(point_in_question)
        print(point_prediction)
        predictions = reg.predict(np.sort(x_index, axis = 0))
        plt.title(stock)
        plt.plot(x_index, predictions, label = "reg predictions")
        plt.plot(x_index, y_index, label= "real")
        plt.scatter([point_in_question], [point_prediction], label = 'stockit.predict[{0}]'.format(point_in_question))
        plt.legend()
        plt.show()

    def moving_avg_demo():
        #call the moving average method of the stockit_class
        plt.title(stock)
        stockit.moving_avg(index = 9,show_plt=False,save_plt=True,name= f'{stock}.png')

    def moving_avg_poly_reg_demo():
        style.use('ggplot')
        stockit.MA_reg(reg_degree=10, MA_index=9, reg_index=300)
        point_in_question = max+1
        point_prediction = stockit.predict(point_in_question)
        print(point_prediction)

        predictions = reg.predict(np.sort(x_poly, axis = 0))
        #x values for graphing

        #x_index = []
        #for current_index_x in range(len(predictions)):
            #x_index.append(current_index_x)

        #true y values for graphing

        y_index = []
        for index_y in range(len(df)):
            y_index.append(df[index_y])
        plt.title(stock)
        plt.plot(x_index, predictions, label = "MA reg predictions")
        plt.plot(x_index, y_index, label= "real")
        plt.scatter([point_in_question], [point_prediction], label = 'stockit.predict[{0}] (MA reg)'.format(point_in_question))
        plt.legend()
        plt.show()


    def stockit_demo():
        style.use('ggplot')
        stockit.train_poly(degree = 10, index=300)
        point_in_question = max+1
        point_prediction = stockit.predict(point_in_question)
        print(point_prediction)

        #creates the x or independed variable

        predictions = reg.predict(np.sort(x_poly, axis = 0))
        plt.title(stock)
        plt.plot(x_index, predictions, label = "polynomial predictions")
        plt.scatter([point_in_question], [point_prediction], label = 'stockit.predict[{0}]'.format(point_in_question))
        stockit.moving_avg(index = 9, show_plt = False)
        stockit.moving_avg(index = 25, show_plt = False)
        stockit.moving_avg(index = 50, show_plt = False)
        stockit.moving_avg(index = 100, show_plt = True)

    linear_regressor_demo()

if __name__ == '__main__':
    main()
