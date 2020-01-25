#import external pandas_datareader library with alias of web
import pandas_datareader as web

#import datetime internal datetime module
#datetime is a Python module
import datetime

def returnData(stock_ticket):
    '''
    returns pandas dataframe containing stock prices from the past year of the 'stock_ticket'
    '''
    #datetime.datetime is a data type within the datetime module
    start = datetime.datetime.today() - datetime.timedelta(days=365)

    end = datetime.date.today()
    # download data, if data not found, raise exception
    try:
        #DataReader method name is case sensitive
        df = web.DataReader(stock_ticket, 'yahoo', start, end)
        return df
    except:
        raise ValueError(f"yahoo finance cannot find ticket named '{stock_ticket}'")



def downloadData(stock_ticket):
    '''
    downloads stock prices from the past year of the 'stock_ticket' and saves it in a file called 'stock_ticket.csv'
    '''
    #datetime.datetime is a data type within the datetime module
    start = datetime.datetime.today() - datetime.timedelta(days=365)

    end = datetime.date.today()
    # download data, if data not found, raise exception
    try:
        #DataReader method name is case sensitive
        df = web.DataReader(stock_ticket, 'yahoo', start, end)
    except:
        raise ValueError("yahoo finance cannot find ticket named '{}'".format(stock_ticket))

    df.to_csv(f'{stock_ticket}.csv')
