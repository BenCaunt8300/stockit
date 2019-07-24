# stockit


This is a project ive been working on in my spare time the past 3 weeks 

The goal is to eventually get the program to work with something like robinhood or stockpile in order to buy and sell stocks (stonks)
in real time 

stockit_class_sklearn.py has a class that is the regressor that uses polynomial regression
 
stockit_main.py takes real time data from yahoo and with a theoretical $50k and buys as much theoretical stock as it can with it 

it will the use that live data from yahoo finance to allow it to calculate a potential climb or fall in stock price and then make appropriate actions on that information


# Change Log

<i>Wednesday, July 24 12:45 am (ADT)</i>
over the past two days ive implemented a few boring general optimizations, things like variables that only need to be definded once and not in a for loop and the relatively resource intensive sklearn linearregression() class running twice for no reason 

after these fixes i implemented the very popular moving average analysis technique with a method called moving_avg()
this method takes in the inputs of self and index

self is used to get the data 
index i used to tell the program how far back you want each moving average window to go 
