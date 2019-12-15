# stockit
<a href="https://codeclimate.com/github/BenCaunt8300/stockit/maintainability"><img src="https://api.codeclimate.com/v1/badges/9c395b17b6a40f82dd61/maintainability" /></a>
[![Downloads](https://pepy.tech/badge/stockit)](https://pepy.tech/project/stockit)
<a href="https://codeclimate.com/github/BenCaunt8300/stockit/test_coverage"><img src="https://api.codeclimate.com/v1/badges/9c395b17b6a40f82dd61/test_coverage" /></a>
# Install: 

Method 1 (pip):
'''
pip3 install stockit
'''

Method 2 (github clone)
step 1: clone the github repo
step 2: within the cloned directory, run the following command:
```
pip3 install .
```

stockit is a python class that aids in easy price estimation and alaysis of stocks

stockit_class.py is the real star of the show here.  It contains a class that has many tools needed for analysis and price estimation of stock or currency prices such as regression and moving average windows.
 
stockit_realtime.py is for experimentation and takes real time data from yahoo and with a theoretical $50k and buys as much theoretical stock as it can with it 

it will the use that live data from yahoo finance to allow it to calculate a potential climb or fall in stock price and then make appropriate actions on that information

stockits regression usage can be demoed here: [bencaunt1232.pythonanywhere.com]
type ```/stockit-app/[stock name]``` to make the estimation

# USAGE:

get stock data with Close/close column from a csv or other file as a pandas dataframe
```python
import pandas as pd 
data = pd.read('example.csv')
```
import the stockit class
```python
from stockit import stockit_class
```

then lets create an instance of stockit_class, passing it our pandas dataframe

```python

stockit = stockit_class(data)

```

from here we can do a few things 

1. we can use the polynomial regression feature 
2. we can use the newly added (as of july 24 2019) moving average feature 

Regression analysis 
```python
 #next day that we will estimate the price of 
 next = len(data)+1
 
 # fit the model to the dataset
 stockit.train(index = 300)
 
 #make estimation on the next day 
 print(stockit.predict(next))

```

moving average
```python

 #simply call the moving_avg() method of stockit
 #index specifies the length of each window, lower = closer fit to live data, higher = smoother line, your choice
 stockit.moving_avg(index = 35)
 
```

Regressuib and moving Average analysis
```python
import pandas as pd 
from stockit_class import stockit_class
data = pd.read_csv('example.csv')
stockit = stockit_class(data)
stockit.train()
print(stockit.predict(100))
stockit.moving_avg(index = 35)

```
