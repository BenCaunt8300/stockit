#import external pandas_datareader library with alias of web
import pandas_datareader as web

#import datetime internal datetime module
#datetime is a Python module
import datetime

DAYS_IN_YEAR = 365
def returnData(stock_ticket, years = 3):
    '''
    returns pandas dataframe containing stock prices from the past year of the 'stock_ticket'

    Years is the number of years of data that is collected
    '''
    #datetime.datetime is a data type within the datetime module
    start = datetime.datetime.today() - datetime.timedelta(days=int(DAYS_IN_YEAR * years))

    end = datetime.date.today()
    # download data, if data not found, raise exception
    try:
        #DataReader method name is case sensitive
        df = web.DataReader(stock_ticket, 'yahoo', start, end)
        return df
    except:
        raise ValueError(f"yahoo finance cannot find ticket named '{stock_ticket}'")



def downloadData(stock_ticket, years = 3):
    '''
    downloads stock prices from the past year of the 'stock_ticket' and saves it in a file called 'stock_ticket.csv'

    Years is the number of years of data that is collected
    '''
    #datetime.datetime is a data type within the datetime module
    start = datetime.datetime.today() - datetime.timedelta(days=int(DAYS_IN_YEAR * years))

    end = datetime.date.today()
    # download data, if data not found, raise exception
    try:
        #DataReader method name is case sensitive
        df = web.DataReader(stock_ticket, 'yahoo', start, end)
    except:
        raise ValueError("yahoo finance cannot find ticket named '{}'".format(stock_ticket))

    df.to_csv(f'{stock_ticket}.csv')
