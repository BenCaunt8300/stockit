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

        #distance back that the neural network compares for each day(n day history in this case)
        index = 10
        x = {}
        y = []

        for z in range(index):
            x['l'+str(z)] = []

        #number of times the n day history is calculated
        intervals = data_max-(index-1)

        #the start is equal to 99 because it counts back from the start ie n-1, n-2, n-3... 
        #TO DO *** fix this comment, i have no idea what this is saying ^^^

        start = index-1
        '''
        the list "entire_past" is going to store each of the past n values
        it will then be used to determine the maximum value which every one of
        the past n will be devided by to be normalized
        the list will then be erased
        '''

        entire_past = []
        for count_1 in tqdm(range(intervals)):
            for count_2 in range(index):
                entire_past.append((index-1)-count_2)
            past_max = max(entire_past)
            for count_len_past in range(len(entire_past)-1):
                #i would like to append
                x['l'+str(count_2)].append(((entire_past[start-count_len_past])/past_max))

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
        #x = np.array(x)
        y = np.array(y)
        print(x.head())
        x.to_csv('stockit_classifier_window.csv')

'''
        #neural network stuff
        tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)

        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Dense(64, kernel_initializer='random_normal', bias_initializer='random_uniform', activation = 'elu'))
        model.add(tf.keras.layers.Dense(64, kernel_initializer='random_normal', bias_initializer='random_uniform', activation = 'elu'))
        model.add(tf.keras.layers.Dense(64, kernel_initializer='random_normal', bias_initializer='random_uniform', activation = 'elu'))
        model.add(tf.keras.layers.Dense(32, kernel_initializer='random_normal', bias_initializer='random_uniform', activation = 'elu'))
        model.add(tf.keras.layers.Dense(16, kernel_initializer='random_normal', bias_initializer='random_uniform', activation = 'elu'))
        model.add(tf.keras.layers.Dense(2, kernel_initializer='random_normal', bias_initializer='random_uniform', activation='sigmoid'))

        model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

        model.fit(x, y, epochs = 20)
        model_json = model.to_json()

        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model.h5")
        print("Saved model to disk")
'''

def main():
    stockit = classifier(df)
    stockit.train()


if __name__ == "__main__":
    main()
 
