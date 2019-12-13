import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from matplotlib.pyplot import style
from statistics import mean
import warnings


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
        print("******************************************************************************")
        print("Disclaimer: Stockit predictions are not investment advice.  They are meerly   ")
        print("information used for educational purposes,  if used for investment, use at    ")
        print("Ones own risk.  Thank you for using stockit.")
        print("******************************************************************************")
        #for determining if the train and predict methods are using polynomials or not
        self.poly_reg_bool = True
        #for polynomial features class from sklearn
        self.poly = None
        self.reg = None
        self.x_index = None # graphing stuff
        self.y_index = None #graphin stuff,

        
    #poly regressor training function
    def train(self, degree = 10, index = 0, poly_bool = False):
        # if the index is greater than the length of the data raise an error because the linear regressor should not properly be able to train
        if index > len(self.data):
            raise ValueError("'index' cannot be greater than the length of your CSV file. ")
        self.poly_reg_bool = poly_bool

        self.reg = LinearRegression()

        #if index is equal to 0 then do things as normally
        if index == 0:
            x = []

            y = self.data
            #creates the x or independed variable
            for i in tqdm(range(len(self.data))):
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
        #only runs if poly_reg_bool is equal to true
        #if so polynomial regression is in use, if not it is linear regression
        if self.poly_reg_bool:
            self.poly = PolynomialFeatures(degree = degree)
            x_poly = self.poly.fit_transform(x)
            self.poly.fit(x_poly, y)
        else:
            #poly_reg_bool is false so using linear regression
            self.reg.fit(x,y)
        return 1
            
    #predict method
    def predict(self, target):
        # if the self.reg object is None, this means that train has not been called, therefore we should just call it anyways
        if self.reg is None:
            warnings.warn("""self.reg == None, this most likely means you did not call the .train() method
            automatically calling train() method
            manually call the train() method for added options""")
            self.train()
        pred = np.array(target)
        pred = pred.reshape(1,-1)
        if self.poly_reg_bool:
            pred_poly = self.poly.fit_transform(pred)
            pred  = pred_poly
            
        
        output = self.reg.predict(pred)
        return output[0]
 
    #moving average, input the number of time stamps with the 'index' variable
    def moving_avg(self, index = 100, show_real = True, show_plt = True, save_plt = False, name = "name", save_index = 90, save_dpi = 800):

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
        x_data_graphing = [i for i in range(len(data))]

        x = []

        #calculate moving average for duration of the argument index

        #list of all the moving average values
        moving_avg_values = []
        #fill the first 'range(index)' with 0s because why not lol
        for fill in range(index):
            moving_avg_values.append(0)

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
                #dont worry about this
                #print("stuff happens, moving on")
                #get out of here lol
                #we've gone as far as we can, stop here, youre wasting CPU time


        #fill in the x values for graphing
        for length_mov_avg_val in range(len(moving_avg_values)):
            x.append(length_mov_avg_val)

        #debug stuff, uncomment if you need lol

        #print(f"len(x) = {len(x)}")
        #print(f"len(moving_avg_values) = {len(moving_avg_values)}")

        if save_plt:
            x = pd.DataFrame(x)
            moving_avg_values = pd.DataFrame(moving_avg_values)
            x_data_graphing = pd.DataFrame(x_data_graphing)
            x = x.tail(save_index)
            data = data.tail(save_index)
            moving_avg_values = moving_avg_values.tail(save_index)
            x_data_graphing = x_data_graphing.tail(save_index)

        plt.plot(x, moving_avg_values, label = f"moving average {index}")

        #the show_real variable is a variable that when true, plots the true stock data
        if show_real:
            plt.plot(x_data_graphing, data, label = "real values")

        if show_plt:
            plt.legend()
            plt.show()
        if save_plt:
            plt.legend()
            plt.savefig(name, dpi = save_dpi)


# basically a bunch of examples
def main():

    #creates pandas dataframe
    stock = 'NVDA'

    df = pd.read_csv("NVDA.csv")
    #the last index of a dataset is equal to its length - ya bois law
    data_len = len(df)
    #prints the length of the dataset
    print("df length is: {0}".format(len(df)))

    stockit = stockit_class(df)


    def linear_regressor_demo():
        style.use('ggplot')
        stockit.train(index = 300, poly_bool=False)
        point_in_question = data_len+1
        point_prediction = stockit.predict(point_in_question)
        print(point_prediction)
        predictions = stockit.reg.predict(np.sort(stockit.x_index, axis = 0))
        plt.title(stock)
        plt.plot(stockit.x_index, predictions, label = "reg predictions")
        plt.plot(stockit.x_index, stockit.y_index, label= "real")
        plt.scatter([point_in_question], [point_prediction], label = 'stockit.predict[{0}]'.format(point_in_question))
        plt.legend()
        plt.show()

    def moving_avg_demo():
        #call the moving average method of the stockit_class
        plt.title(stock)
        stockit.moving_avg(index = 9, show_plt=False, save_plt=True, name= f'{stock}.png')

    def stockit_demo():
        style.use('ggplot')
        stockit.train(degree = 10, index=300, poly_bool=False)
        point_in_question = data_len+1
        point_prediction = stockit.predict(point_in_question)
        print(point_prediction)


        #creates the x or independed variable

        predictions = stockit.reg.predict(np.sort(stockit.x_index, axis = 0))


        plt.title(stock)
        plt.plot(stockit.x_index, predictions, label = "reg predictions")
        plt.scatter([point_in_question], [point_prediction], label = f'stockit.predict[{point_in_question}]')
        stockit.moving_avg(index = 100, show_plt = True)

    linear_regressor_demo()

if __name__ == '__main__':
    main()
