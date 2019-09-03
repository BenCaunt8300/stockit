'''
* stockit_main.py
*purpose:
    import the class from the file stockit_class_sklearn.py
    then use this class to create predictions on stock information updating in near real time
    Real time stock data will come from the library yahoo_fin
'''

#real time stock data
from yahoo_fin import stock_info as si

#our custom class from stockit_class_sklearn.pys
from stockit_class_sklearn import stockit_class
#used for waiting and stuff
import time
#matrix math and stuff
import numpy as np
#graphing stuff
import matplotlib.pyplot as plt

#creates empty python list that our data will be appended to
data = []

#stockit value trough the day
stockit_value = []

#holding value through the day
hold_value_lst = []

#stock symbol
stock_symbol = "AMD"

#amount of money to start
money = 50000

hold_money = 50000
hold_start = hold_money
start = money
#original buying price
orig_price = si.get_live_price(stock_symbol)

#shares bought, whole shares not parts
shares = money/orig_price

#holding shares, used for comparison
holding_shares = hold_money/si.get_live_price(stock_symbol)

#leftover money
money = money - orig_price*shares
#leftover holding money
hold_money = hold_money - orig_price*holding_shares

#find current value
def total_value():
    share_value = shares * si.get_live_price(stock_symbol)
    return share_value + money

def holding_value():
    holding_value = holding_shares * si.get_live_price(stock_symbol)
    return holding_value + hold_money

#has stockit bought and is currently holding the price?
holding = True

def buy():
    global shares
    global holding
    global money
    shares = money/si.get_live_price(stock_symbol)
    money = money - (si.get_live_price(stock_symbol)*shares)
    holding = True

def sell():
    global money
    global shares
    global holding
    #basically re-declare the money variable as the current value of the shares + the current leftover balance
    money = total_value()
    shares = 0
    holding = False

def trade():
    #initiate stockit
    stockit = stockit_class(data)

    #at the start of the program, it collects data for the first hour then makes predictions on it
    for i in range(10):
        data.append(si.get_live_price(stock_symbol))
        #wait for 60 seconds
        #wait(60)
    #inital training of stockit
    stockit.train(degree = 10)

    for i in range(50):
        try:
            #have stockit make a prediction on what the next price will be
            prediction = stockit.predict(len(data)+1)
            #the current last price
            current_last_price = data[-1]
            #actually find that price
            current_price = si.get_live_price(stock_symbol)
            data.append(si.get_live_price(stock_symbol))
            if current_last_price < prediction and holding == False:
                print("stockit predicts a bull! now buying!!!")
                buy()
            elif current_price > prediction and holding == True:
                print("stockit predicts a bear... now selling!!!")
                sell()
            else:
                #do nothing
                pass

            #retrain stockit with new data
            current_value = total_value()
            stockit_value.append(current_value/500)
            hold_value_current = holding_value()
            hold_value_lst.append(hold_value_current/500)
            print(f"current_value = {current_value}")
            print(f"holding_value_current = {hold_value_current}")
            #if the list gets too long, make use of stockits indexing feature
            if len(data) > 30:
                stockit.train(degree = 10, index=10)
            else:
                stockit.train(degree = 10)
            time.sleep(1)
        except:
            pass

def main():
    trade()
    final_total = total_value()
    hold_final = holding_value()
    print(f"stockit started with {start}")
    print(f"stockit ended with {final_total}")

    print(f"holding started with {hold_money}")
    print(f"holding ended with {hold_final}")
    #x_data, x_stockit, x_hold used for graphing
    x_data = []
    x_stockit = []
    x_hold = []

    for data_x_vals in range(len(data)):
        x_data.append(data_x_vals)
    for stockit_x_vals in range(len(stockit_value)):
        x_stockit.append(stockit_x_vals)

    for holding_x_vals in range(len(hold_value_lst)):
        x_hold.append(holding_x_vals)
    plt.plot(x_data,data, label="price trough the day ")
    plt.plot(x_stockit, stockit_value, label = "stockit value (normalized, devided by 500)")
    plt.plot(x_hold, hold_value_lst, label = "holding value (normalized)")
    plt.title(f"{stock_symbol} today")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
