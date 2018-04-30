import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import sklearn.preprocessing as skp
import numpy
from sklearn.metrics import mean_squared_error
from scipy.ndimage.interpolation import shift

def string_convert(text):
    return text.split()[0]

def date_convert(string):
    return datetime.strptime(string, '%Y-%m-%d')

df = pd.read_csv('grocery_transactions.csv',header=0,names=['acc', 'date', 'alpha_text', 'amount'], dtype={0:np.int32, 3:np.float64},engine='python', converters={'alpha_text':string_convert, 'date':date_convert})

df2 = df[df.acc == 4680070].sort_values(by=['date'], ascending=True)

df3 = df2[df2.alpha_text == "coop"].sort_values(by=['date'], ascending=True)

df4 = df3.filter(['date','amount'], axis=1)

def prepare_data(df, events, timesteps):

    raw_values = df.values

    tte_series = difference(raw_values, 1)
    tte_values = tte_series.values

    tte_values = [v[0] for v in tte_values]
    tte_values.insert(0,0)

    event_amount = [v[1] for v in raw_values]



def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        value[0] = value[0].days
        # print(isinstance(dataset[i], np.array))
        # print(isinstance(value[0], timedelta))
        diff.append(value)
    return pd.Series(diff)



