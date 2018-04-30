#%matplotlib inline
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import wtte.weibull as weibull
import wtte.wtte as wtte
from wtte.wtte import WeightWatcher

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,GRU
from keras.layers import Lambda
from keras.layers.wrappers import TimeDistributed
import sklearn.preprocessing as skp
from sklearn.metrics import mean_squared_error
from scipy.ndimage.interpolation import shift

from keras.optimizers import RMSprop,adam
from keras.callbacks import History




def string_convert(text):
    return text.split()[0]

def date_convert(string):
    return datetime.strptime(string, '%Y-%m-%d')


df = pd.read_csv('../../data/grocery_transactions.csv',header=0,names=['acc', 'date', 'alpha_text', 'amount'], dtype={0:np.int32, 3:np.float64},engine='python', converters={'alpha_text':string_convert, 'date':date_convert})
df.head()

df2 = df[df.acc == 4680070].sort_values(by=['date'], ascending=True)

df3 = df2[df2.alpha_text == "coop"].sort_values(by=['date'], ascending=True)

df4 = df3.filter(['date','amount'], axis=1)



def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        value[0] = value[0].days
        # print(isinstance(dataset[i], np.array))
        # print(isinstance(value[0], timedelta))
        diff.append(value)
    return pd.Series(diff)



df4.set_index('date', inplace=True)

df4 = df4[~df4.index.duplicated(keep='first')]

print("Duplicates:",df4[df4.index.duplicated()])

df4.index = pd.DatetimeIndex(df4.index)

df4 = df4.reindex(index = pd.date_range('03-24-2013', '02-18-2017'), fill_value=0)

df4 = df4.amount.apply(lambda x: 1. if x>0 else 0.)

### example

idx = pd.date_range('09-01-2013', '09-30-2013')

s = pd.Series({'09-02-2013': 2,
               '09-03-2013': 10,
               '09-06-2013': 5,
               '09-07-2013': 1})
s.index = pd.DatetimeIndex(s.index)

s = s.reindex(idx, fill_value=0)

#add days at 90 days intervals

test = df4[0:480]

#test.reset_index()
test = test.reset_index()

test = test.values

train_y = df4.reset_index()
raw_values = df4.values

train = raw_values
#test = raw_values[1100:]

train_y = train_y.values

x, y = train[:714], train_y[:714]

x_list = []
y_list = []
for i in range(17):
    j = i+1
    x_list.append(x[i*42:j*42])
    y_list.append(y[i*42:j*42])

def get_int_from_date(y_val):
    date = y_val[0]
    return [(date - datetime.strptime('03-24-2013', '%m-%d-%Y')).days, y_val[1]]

x_train = np.array([[[x] for x in n] for n in x_list])
y_train = np.array([[get_int_from_date(y) for y in n] for n in y_list])


x_t, y_t = test[:240], test[:240]

x_tlist = []
y_tlist = []

for i in range(5):
    j=i+1
    x_tlist.append(x_t[i*42:j*42])
    y_tlist.append(y_t[i*42:j*42])

x_test = np.array([[[x[1]] for x in n] for n in x_tlist])
y_test = np.array([[get_int_from_date(y) for y in n] for n in y_tlist])

print('x:',x_train.shape, x_test.shape)
print('y:',y_train.shape, y_test.shape)



tte_mean_train = np.nanmean(train[:,])
init_alpha = -1.0/np.log(1.0-1.0/(tte_mean_train+1.0) )
mean_u = np.nanmean(train[:,])
init_alpha = init_alpha/mean_u
print('init_alpha: ',init_alpha,'mean uncensored: ',mean_u)

history = History()
weightwatcher = WeightWatcher()

n_features = 1

# Start building the model
model = Sequential()
# Dont need to specify input_shape=(n_timesteps, n_features) since keras uses dynamic rnn by default

#model.add(LSTM(1, input_shape=(714,), activation='tanh', return_sequences=True))
model.add(GRU(1,input_shape=x_train.shape[1:],activation='tanh',return_sequences=True))

model.add(Dense(2))
model.add(Lambda(wtte.output_lambda,
                 arguments={"init_alpha":init_alpha,
                            "max_beta_value":4.0}))
loss = wtte.loss(kind='discrete').loss_function

model.compile(loss=loss, optimizer=adam(lr=.01))
# model.compile(loss=loss, optimizer=adam(lr=.01),sample_weight_mode='temporal') # If varying length

model.summary()


model.fit(x_train,y_train,
          epochs=60,
          batch_size=1,
          verbose=1,
          validation_data=(x_test, y_test),
#           sample_weight = sample_weights # If varying length
          callbacks=[history,weightwatcher])



# plt.plot(history.history['loss'],    label='training')
# plt.plot(history.history['val_loss'],label='validation')
# plt.title('loss')
# plt.legend()
#
# weightwatcher.plot()
#
# # Make some parametric predictions
# print('TESTING (no noise in features)')
# print('(each horizontal line is a sequence)')
predicted = model.predict(x_test)
# print(predicted.shape)
#
# plt.imshow(predicted[:,:,0],interpolation="none",cmap='jet',aspect='auto')
# plt.title('predicted[:,:,0] (alpha)')
# plt.colorbar(orientation="horizontal")
# plt.show()
# plt.imshow(predicted[:,:,1],interpolation="none",cmap='jet',aspect='auto')
# plt.title('predicted[:,:,1] (beta)')
# plt.colorbar(orientation="horizontal")
# plt.show()
#
# print('TRAINING (Noisy features)')
# predicted = model.predict(x_train[:,])
# print(predicted.shape)
#
# plt.imshow(predicted[:,:,0],interpolation="none",cmap='jet',aspect='auto')
# plt.title('predicted[:,:,0] (alpha)')
# plt.colorbar(orientation="horizontal")
# plt.show()
# plt.imshow(predicted[:,:,1],interpolation="none",cmap='jet',aspect='auto')
# plt.title('predicted[:,:,1] (beta)')
# plt.colorbar(orientation="horizontal")
# plt.show()

from wtte.plots.weibull_heatmap import weibull_heatmap
# TTE, Event Indicator, Alpha, Beta
drawstyle = 'steps-post'

print('one training case:')
print('Cautious (low beta) until first signal then almost thrown off track by noise')
every_nth = 17
batch_indx = int(2+every_nth//4)

# Pick out data for
# one sequence
print('numpy array of predictions:',predicted)
n_timesteps = 42
this_seq_len = n_timesteps
a = predicted[batch_indx,:this_seq_len,0]
b = predicted[batch_indx,:this_seq_len,1]
t = np.array(range(len(a)))
x_this = x_train[batch_indx:this_seq_len,:]

tte_censored = y_train[batch_indx:this_seq_len,0]
tte_actual   = y_test[batch_indx:this_seq_len,0]

u = y_train[batch_indx,:this_seq_len,1]>0
##### Parameters
# Create axes
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(t, a, color='b')
ax1.set_xlabel('time')
ax1.set_ylabel('alpha')

ax2.plot(t, b, color='r')
ax2.set_ylabel('beta')

# Change color of each axis
def color_y_axis(ax, color):
    """Color your axes."""
    for t in ax.get_yticklabels():
        t.set_color(color)
    return None
color_y_axis(ax1, 'b')
color_y_axis(ax2, 'r')
plt.show()

##### Prediction (Using weibull-quantities like quantiles etc)
plt.plot(tte_censored,label='censored tte',color='black',linestyle='dashed',linewidth=2,drawstyle=drawstyle)
plt.plot(t,tte_actual,label='uncensored tte',color='black',linestyle='solid',linewidth=2,drawstyle=drawstyle)

plt.plot(weibull.quantiles(a,b,0.75),color='blue',label='pred <0.75',drawstyle=drawstyle)
plt.plot(weibull.mode(a, b), color='red',linewidth=1,label='pred mode/peak prob',drawstyle=drawstyle)
plt.plot(weibull.mean(a, b), color='green',linewidth=1,label='pred mean',drawstyle='steps-post')
plt.plot(weibull.quantiles(a,b,0.25),color='blue',label='pred <0.25',drawstyle=drawstyle)

plt.xlim(0, this_seq_len)
plt.xlabel('time')
plt.ylabel('time to event')
plt.title('prediction sequence '+str(batch_indx),)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

##### Prediction (Density)
fig, ax = plt.subplots(1)

fig,ax = weibull_heatmap(
    fig,ax,
    t,
    a,
    b,
    max_horizon = int(1.5*tte_actual.max()),
    time_to_event=tte_censored,
    true_time_to_event=tte_actual,
    censoring_indicator = ~u,
    title='predicted Weibull pmf $p(t,s)$ sequence '+str(batch_indx),
    lw=3.0,
    is_discrete=True,
    resolution=None,
    xax_nbins=10,
    yax_nbins=4
)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
