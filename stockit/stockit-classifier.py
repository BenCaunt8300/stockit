import tensorflow as tf
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import random

scaler = MinMaxScaler()

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
        index = 350
        x = {}
        y = []

        for z in range(index):
            x['l'+str(z)] = []

        #number of times the n day history is calculated
        intervals = data_max-(index-1)

        #where we start incrementing
        start = index
        '''
        the list "entire_past" is going to store each of the past n values
        it will then be used to determine the maximum value which every one of
        the past n will be devided by to be normalized
        the list will then be erased
        '''


        #increment through the length of 'intervals'
        for count_1 in tqdm(range(intervals)):
            #store the next start position as a variable
            next = start + 1
            #append each value from the index to the the list past_index
            #runs for how long the int index is set to
            #scale each appended value between 0 and 1 by deviding past_index by the maximum value in the list and append those scaled values to the dictionary
            past_index = []
            #append to list past_index
            for count_2 in range(index):

                past_index.append(data[(start-1)-count_2])

            #scale past_index between 0-1
            past_index_df = pd.DataFrame(past_index)
            past_index_scaled = scaler.fit_transform(past_index_df)

            for count_3 in range(index):
                x['l'+str(count_3)].append(past_index_scaled[count_3])
            try:
                if data[start] <= data[next]:
                    y.append(1)
                elif data[start] > data[next]:
                    y.append(0)
            except:
                print("except hit")
                y.append(0)
            #incremnt start up by 1
            start += 1
        x = pd.DataFrame(x)
        print(x.head())
        y = pd.DataFrame(y)
        first_80percent = round(len(x)*0.8)
        last_20percent = round(len(x)*0.2)
        x_train = x.head(first_80percent)
        y_train = y.head(first_80percent)
        x_test = x.tail(last_20percent)
        y_test = y.tail(last_20percent)
        #x.to_csv('stockit_classifier_window_x.csv')
        y.to_csv('stockit_classifier_window_y.csv')

        x_train = np.array(x_train)
        x_train = np.expand_dims(x_train, axis = 2)
        #len_x = len(x)
        #x = x.reshape(len_x,)
        y_train = np.array(y_train)

        x_test = np.array(x_test)
        x_test = np.expand_dims(x_test, axis = 2)
        #len_x = len(x)
        #x = x.reshape(len_x,)
        y_test = np.array(y_test)

        '''
        x, y = np.arange(10).reshape((5,2)), range(5)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
        '''
        #neural network stuff
        tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv1D(64, kernel_size = 3, activation = 'relu', kernel_initializer='random_normal', bias_initializer='random_uniform', input_shape = (index, 1) ))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model.add(tf.keras.layers.Conv1D(32, kernel_size = 3, kernel_initializer='random_normal', bias_initializer='random_uniform', activation = 'relu' ))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, kernel_initializer='random_normal', bias_initializer='random_uniform', activation = 'relu'))
        model.add(tf.keras.layers.Dense(32, kernel_initializer='random_normal', bias_initializer='random_uniform', activation = 'relu'))
        model.add(tf.keras.layers.Dense(16, kernel_initializer='random_normal', bias_initializer='random_uniform', activation = 'elu'))
        model.add(tf.keras.layers.Dense(2, kernel_initializer='random_normal', bias_initializer='random_uniform', activation='elu'))

        model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

        model.fit(x_train, y_train, batch_size=520, epochs = 100)
        model.evaluate(x_test, y_test)
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
