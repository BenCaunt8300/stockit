# stockit
<a href="https://codeclimate.com/github/BenCaunt8300/stockit/maintainability"><img src="https://api.codeclimate.com/v1/badges/9c395b17b6a40f82dd61/maintainability" /></a>
[![Downloads](https://pepy.tech/badge/stockit)](https://pepy.tech/project/stockit)
# Install: 

Method 1 (pip):

```
pip3 install stockit
```

Method 2 (github clone)

step 1: clone the github repo

step 2: within the cloned directory, run the following command:
```
pip3 install .
```

stockit is a python module that aids in easy price estimation and alaysis of stocks

Stockit is a python object that on initailization requires a Pandas dataframe containing historic stock prices.  
Stockit is designed to work with the close price and searches for a close column in the dataframe

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

from here we can do a few things:
1. Simple linear Regression
2. Support Vector Regression
3. Plot Moving Average


Regression analysis:
```python
 #next day that we will estimate the price of 
 next = len(data)+1
 
 # fit the model to the dataset
 stockit.train(index = 300)
 
 #make estimation on the next day 
 print(stockit.predict(next))

```
stockit can also use support vector regression to achieve a tighter fit to the data:
```python 

 stockit.train(index = 300, SVRbool = True)
 print(stockit.predict(next))

```

moving average:
```python

 #simply call the moving_avg() method of stockit
 #index specifies the length of each window, lower = closer fit to live data, higher = smoother line, your choice
 stockit.moving_avg(index = 35)
 
```

Regression and Moving Average Analysis
```python
import pandas as pd 
from stockit_class import stockit_class
data = pd.read_csv('example.csv')
stockit = stockit_class(data)
stockit.train()
print(stockit.predict(100))
stockit.moving_avg(index = 35)

```
