import tensorflow as tf
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
stock = 'NVDA'
df = pd.read_csv(f'{stock}.csv')

class classifier():
    def __init__(self, data):
        self.data = data
    def train(self):
        #copy pandas dataframe
        data = self.data
        data = data.close
        data = np.array(data)
        data_max = len(data)

        #distance back that the neural network compares for each day(100 day history in this case)
        history_int = 100
        x = {}
        y = []

        for z in range(100):
            x['l'+str(z)] = []

        #number of times the 100 day history is calculated
        intervals = data_max-99

        #the start is equal to 99 because it counts back from the start ie 99,98...1,0 to get to 100
        start = 99
        '''
        the list "entire_past_100" is going to store each of the past 100 values
        it will then be used to determine the maximum value which every one of
        the past 100 will be devided by to be normalized
        the list will then be erased
        '''

        entire_past_100 = []
        for count_1 in tqdm(range(intervals)):
            for count_2 in range(100):
                entire_past_100.append(99-count_2)
            past_max = max(entire_past_100)
            for count_len_past_100 in range(len(entire_past_100)-1):
                #i would like to append
                x['l'+str(count_2)].append(((entire_past_100[start-count_len_past_100])/past_max))

            #the neural network will work by calculating wheather the price will
            #go up or down NOT an actual price, we do this with the following code
            #0 for down, 1 for up or stay

            #right after the past 100 is this value
            next = start+1

            #print(data[start])
            try:
                if data[next] >= data[start]:
                    y.append(1)
                elif data[next] < data[start]:
                    y.append(0)
                else:
                    print('error,  this shouldnt happen, lern 2 code plz')
            except:
                #just ignore this lol, its only one time to fix a no bueno issue
                y.append(random.choice([0,1]))
                #print("you reached the except you dummy")
            #increment the start up by one
            start += 1
        x = pd.DataFrame(list(x.items()))
        x = np.array(x)
        y = np.array(y)

        #neural network stuff
        tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)

        model = tf.keras.models.Sequential()

        #add model layers
        model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))

        model.add(tf.keras.layers.Conv2D(32, kernel_initializer='random_normal',bias_initializer='random_uniform', kernel_size=3, activation='relu'))

        model.add(tf.keras.layers.Flatten())

        for i in range(2):
            model.add(tf.keras.layers.Dense(16, kernel_initializer='random_normal', bias_initializer='random_uniform', activation = 'relu'))

        model.add(tf.keras.layers.Dense(2, kernel_initializer='random_normal', bias_initializer='random_uniform', activation='sigmoid'))

        model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

        model.fit(x, y, epochs = 20)
        model_json = model.to_json()

        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model.h5")
        print("Saved model to disk")


def main():
    stockit = classifier(df)
    stockit.train()


if __name__ == "__main__":
    main()
