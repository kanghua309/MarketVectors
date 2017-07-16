import pandas as pd
import os

datapath = './daily'
filepath = os.path.join(datapath,os.listdir('./daily')[0])

import re
ticker_regex = re.compile('.+_(?P<ticker>.+)\.csv')
get_ticker =lambda x :ticker_regex.match(x).groupdict()['ticker']
print(filepath,get_ticker(filepath))