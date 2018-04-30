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

def entropy(p, k):
    ent = 0
    for i in range(k):
        try:
            ent += -p[i]*math.log(p[i])
        except IndexError:
            continue
    return ent

def trading_entropy(dataframe, account):

    df = dataframe[dataframe.acc == account].sort_values(by=['date'], ascending=True)

    merchant_type = ['kiwi', 'bunnpris', 'rema', 'coop', 'meny']

    proportion = df.groupby('alpha_text').size()
    print(proportion)

    t_entropy = entropy(proportion, len(merchant_type))

    return t_entropy

#print(trading_entropy(df, 4652591))
#4680070 churner
#4655952

#print(df.head())

#def prepare_data(df):

acc_df = df['acc']

unique_acc = df['acc'].unique()

df2 = df[df.acc == 4680070].sort_values(by=['date'], ascending=True)

df3 = df2[df2.alpha_text == "coop"].sort_values(by=['date'], ascending=True)

df4 = df3.filter(['date','amount'], axis=1)

# print(df2.head())
# print(df3.head())
# print(df4.head())
#df4.plot(x='date',y='amount')


def series_to_supervised(df, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(df) is list else df.shape[1]
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        #cols.append(shift(df,i,cval=np.NaN))
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        #cols.append(shift(df,-i,cval=np.NaN))
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together

    agg1 = pd.Series(cols)
    agg2 = pd.DataFrame(names)
    aggs = [agg1, agg2]
    agg = pd.concat(aggs, axis=1)




    # droprows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
        #pd.value_counts(agg.values.flatten())

    return agg

print(series_to_supervised(df4))

def plot_forecasts(series, forecasts, n_test):
    # plot the entire dataset in blue
    pyplot.plot(series.values)
    # plot the forecasts in red
    for i in range(len(forecasts)):
        off_s = len(series) - n_test + i - 1
        off_e = off_s + len(forecasts[i]) + 1
        xaxis = [x for x in range(off_s, off_e)]
        yaxis = [series.values[off_s]] + forecasts[i]
        pyplot.plot(xaxis, yaxis, color='red')
    # show the plot
    pyplot.show()

def make_forecasts(model, n_batch, train, test, n_lag, n_seq):
    forecasts = list()
    print(test,len(test))
    for i in range(len(test)):
        X, y = test[i, 0:n_lag], test[i, n_lag:]
        #print(X)
        # make forecast
        forecast = forecast_lstm(model, X, n_batch)
        # store the forecast
        forecasts.append(forecast)
    return forecasts

def persistence(last_ob, n_seq):
    return [last_ob for i in range(n_seq)]

def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        value[0] = value[0].days
        # print(isinstance(dataset[i], np.array))
        # print(isinstance(value[0], timedelta))
        diff.append(value)
    return pd.Series(diff)

def prepare_data(series, n_test, n_lag, n_seq):
    # extract raw values

    raw_values = series.values

    # transform data to be stationary
    diff_series = difference(raw_values, 1)
    diff_values = diff_series.values

    diff_values = [v[0] for v in diff_values]
    # diff_values = diff_values.reshape(len(diff_values), 1)
    # rescale values to -1, 1
    scaler = skp.MinMaxScaler(feature_range=(-1,1))
    scaled_values = scaler.fit_transform(diff_values)
    scaled_values = scaled_values.reshape(len(scaled_values),1)
    # transform into supervised learning problem X, y
    supervised = series_to_supervised(scaled_values, n_lag, n_seq)
    supervised_values = supervised.values
    # split into train and test sets
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return scaler, train, test

def fit_lstm(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
    # reshape training into [samples, timesteps, features]
    X, y = train[:, 0:n_lag], train[:, n_lag:]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    # design network
    model = Sequential()
    model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # fit network
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
        model.reset_states()
    return model

def inverse_transform(series, forecasts, scaler, n_test):
    inverted = list()
    for i in range(len(forecasts)):
        # create array from forecast
        forecast = numpy.array(forecasts[i])
        forecast = forecast.reshape(1, len(forecast))
        # invert scaling
        inv_scale = scaler.inverse_transform(forecast)
        inv_scale = inv_scale[0, :]
        # invert differencing
        index = len(series) - n_test + i - 1
        last_ob = series.values[index]
        inv_diff = inverse_difference(last_ob, inv_scale)
        # store
        inverted.append(inv_diff)
    return inverted

def inverse_difference(last_ob, forecast):
    # invert first forecast
    inverted = list()
    inverted.append(forecast[0] + last_ob)
    # propagate difference forecast using inverted first value
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i-1])
    return inverted

def evaluate_forecasts(test, forecasts, n_lag, n_seq):
    for i in range(n_seq):
        actual = [row[i] for row in test]
        predicted = [forecast[i] for forecast in forecasts]
        rmse = math.sqrt(mean_squared_error(actual, predicted))
        print('t+%d RMSE: %f' % ((i+1), rmse))

def forecast_lstm(model, X, n_batch):
	# reshape input pattern to [samples, timesteps, features]

    #Y = pd.Series.to_frame(X)
    #Y = Y.reshape(1, 1, len(Y))
    X = X.reshape(1, 1, len(X))

    # make forecast
    forecast = model.predict(X, batch_size=n_batch)
    # convert to array
    return [x for x in forecast[0, :]]



n_lag = 1
n_seq = 3
n_test = 10
n_epochs = 1500
n_batch = 1
n_neurons = 1
# prepare data
scaler, train, test = prepare_data(df4, n_test, n_lag, n_seq)
# fit model
model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)

# make forecasts
forecasts = make_forecasts(model, n_batch, train, test, n_lag, n_seq)

# inverse transform forecasts and test
forecasts = inverse_transform(df4, forecasts, scaler, n_test+2)
actual = [row[n_lag:] for row in test]
actual = inverse_transform(df4, actual, scaler, n_test+2)
# evaluate forecasts
evaluate_forecasts(actual, forecasts, n_lag, n_seq)
# plot forecasts
plot_forecasts(df4, forecasts, n_test+2)