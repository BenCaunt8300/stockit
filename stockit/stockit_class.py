import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from matplotlib.pyplot import style
from stockit import SSRR
from statistics import mean
import warnings

class stockit_class():
    '''The stockit class is used to analyize time series data with its set of algorithms and methods. Pass in pandas dataframe on init'''
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
        print("******************************************************************************")
        print("Disclaimer: Stockit predictions are not investment advice.  They are meerly   ")
        print("information used for educational purposes,  if used for investment, use at    ")
        print("Ones own risk.  Thank you for using stockit.")
        print("******************************************************************************")
        # linear or svr regressor
        self.reg = None
        self.x_index = None
        self.y_index = None
        self.customTrain = None

    def train(self, index = 0, SVRbool = False, SSRRbool = False):
        '''This method fits the linear regression model to the pandas dataframe.
        Index is the number of items starting from the end of the dataset model.

        * SVRbool is a boolean that when true changes the mode to support vector regression and in some cases can have a better fit
        * you may need to play with index to get it just right

        * SSRRbool when true uses the stockit custom serialized slope randomized regression algorithm
        * overidden by SVRbool
        '''
        # determines if using sklearn model or custom model
        if SSRRbool == False:
            self.customTrain = False
        elif SSRRbool == False or SVRbool == True:
            self.customTrain = False
        elif SSRRbool == True:
            self.customTrain = True



        # if the index is greater than the length of the data raise an error because the linear regressor should not properly be able to train
        if index > len(self.data):
            raise ValueError("'index' cannot be greater than the length of your CSV file. ")
        # use support vector regression
        if SVRbool:
            # fixes weird bug where if the index is too long you get an axis error from sklearn
            # set the index to 500 if it is too big.
            if index == 0 or index > 500:
                warnings.warn("ocasionally a strang bug occurs with SVR when the index is set to 0 or is greater than around 500, setting index to 500...")
                index = 500

            self.reg = SVR(kernel='rbf', C=1e1, gamma=0.1)
        # use SSRR algorithm
        elif SSRRbool:
            self.reg = SSRR()
        # if no algorithm specified use linear regression
        else:
            self.reg = LinearRegression()
        #if index is equal to 0 then do things as normally
        if index == 0:

            y = self.data
            #creates x value for graping

            x = [i for i in tqdm(range(len(self.data)))]

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
            #our new x and y data that is just data from the index if it does not equal 0
            x_lst = []
            y_lst = []

            y = self.data

            data_len = len(y)
            for i in tqdm(range(index)):
                #the maximum index is equal to the data length

                distance_back = index-i
                x_lst.append(data_len - distance_back)
            try:
                y_lst = y.tail(index)
            except:
                y_lst = pd.DataFrame(y)
                y_lst = y_lst.tail(index)


            self.x_index = np.array(x_lst)
            self.y_index = np.array(y_lst)

            #reshape data
            self.x_index = self.x_index.reshape(-1,1)
            self.y_index = self.y_index.reshape(-1,1)

            x = self.x_index
            y = self.y_index
            #creates object from sklearn's LinearRegression() class
            #can be called outside the class with stockit_class.reg


        # if using sklearn model
        if self.customTrain == False:
            self.reg.fit(x,y)
        else:
            #print(y.shape)
            #print(y)
            #print(f"y len is {len(y)}")
            self.reg.fit(y)
        return 1

    def predict(self, target):
        '''Use linear regression model that was initalized in the .train() method.  target is the index you are predicting'''
        # if the self.reg object is None, this means that train has not been called, therefore we should just call it anyways
        if self.reg is None:
            warnings.warn("""self.reg == None, this most likely means you did not call the .train() method
            automatically calling train() method
            manually call the train() method for added options""")
            self.train()
        if self.customTrain == False:
            pred = np.array(target)
            pred = pred.reshape(1,-1)
        else:
            pred = target


        output = self.reg.predict(pred)
        return output

    def moving_avg(self, index = 100, show_real = True, show_plt = True, save_plt = False, name = "name", save_index = 90, save_dpi = 800):

        '''
        Calculates and graphs moving average given a specified index or 100 by default.
        '''

        style.use("ggplot")

        #always document your code kids
        #oh yea, this is some moving average thing lol
        #it goes back x days, finds the average, graphs it

        #basically this function takes in the input of a list and finds the average of it,
        #the only difference from the standard mean function is it has the optimization of not having to calculate the length of the datset each time
        #the length of the data that is being average is decided by the index variable
        data = self.data

        #for graphing the real price i think lol
        x_data_graphing = [i for i in range(len(data))]

        #calculate moving average for duration of the argument index

        #list of all the moving average values
        #fill the first 'range(index)' with 0s to graph the first part of the moving average where the full average cannot be calculated
        moving_avg_values = [0 for fill in range(index)]


        '''
        here is where we calculate the moving average for every 'window' of the dataset
        basically we start with the counter variable 'z' + index to get the starting position
        then we go back and average the past 20 positions from the starting variable and then save it to the list
        'moving_avg_values'
        '''

        for z in tqdm(range(len(data))):
            #start 20 after the start of the datset
            current_pos = z+index
            #holds the values of every 20 data points
            try:
                index_values = []
                for y in range(index):
                    #print(f"current_pos-x == {current_pos-y}")
                    index_values.append(data[current_pos-y])
                #print(f"mean(index_values) == {mean(index_values)} ")
                moving_avg_values.append(mean(index_values))
            except:
                pass

        #fill in the x values for graphing

        x = [length_mov_avg_val for length_mov_avg_val in range(len(moving_avg_values))]

        # saves figure to disk if save_plt is set to True
        if save_plt:
            x = pd.DataFrame(x)
            moving_avg_values = pd.DataFrame(moving_avg_values)
            x_data_graphing = pd.DataFrame(x_data_graphing)
            x = x.tail(save_index)
            data = data.tail(save_index)
            moving_avg_values = moving_avg_values.tail(save_index)
            x_data_graphing = x_data_graphing.tail(save_index)

        plt.plot(x, moving_avg_values, label = f"moving average {index}")

        #When true plot the stock data,  disable if plotting many moving average instances ontop of eachother
        if show_real:
            plt.plot(x_data_graphing, data, label = "real values")

        if show_plt:
            plt.legend()
            plt.show()
        if save_plt:
            plt.legend()
            plt.savefig(name, dpi = save_dpi)

    def plotData(self,show_plt = True,save_plt = True, name = "", save_dpi = 900):
        '''
        simply plots the data of the pandas dataframe 'self.data' using matplotlib.pyplot

        method can
        1. Display plot on screen
        2. Save plot to disk
        '''

        # generate plot
        plt.plot(self.data,label = f"{name} Price")

        # y axis label
        plt.ylabel("Price")
        plt.xlabel("Date")
        # show plot
        if show_plt:
            plt.legend()
            plt.show()
        # save plot to disk
        if save_plt:
            plt.legend()
            plt.savefig(name, dpi = save_dpi)




# basically a bunch of examples of how to use the stockit class
def main():

    #creates pandas dataframe
    stock = 'TSLA'
    df = pd.read_csv("TSLA.csv")
    df = df.Close
    #the last index of a dataset is equal to its length - ya bois law
    data_len = len(df)
    #prints the length of the dataset
    print(f"df length is: {len(df)}")

    stockit = stockit_class(df)


    def linear_regressor_demo():
        style.use('ggplot')
        stockit.train()
        point_in_question = data_len+1
        point_prediction = stockit.predict(point_in_question)
        print(point_prediction)
        predictions = stockit.reg.predict(np.sort(stockit.x_index, axis = 0))
        plt.title(stock)
        plt.plot(stockit.x_index, predictions, label = "reg predictions")
        plt.plot(stockit.x_index, stockit.y_index, label= "real")
        plt.scatter([point_in_question], [point_prediction], label = f'stockit.predict[{point_in_question}]')
        plt.legend()
        plt.show()

    def moving_avg_demo():
        #call the moving average method of the stockit_class
        plt.title(stock)
        stockit.moving_avg(index = 9, show_plt=False, save_plt=True, name= f'{stock}.png')

    def stockit_demo():
        style.use('ggplot')
        stockit.train(index=250)
        point_in_question = data_len+1
        point_prediction = stockit.predict(point_in_question)
        print(point_prediction)

        predictions = stockit.reg.predict(np.sort(stockit.x_index, axis = 0))


        plt.title(stock)
        plt.plot(stockit.x_index, predictions, label = "reg predictions")
        plt.scatter([point_in_question], [point_prediction], label = f'stockit regression of day: {point_in_question}')
        stockit.moving_avg(index = 25, show_plt = False)
        plt.savefig("stockit example.png",dpi=1200)


    stockit_demo()

if __name__ == '__main__':
    main()
