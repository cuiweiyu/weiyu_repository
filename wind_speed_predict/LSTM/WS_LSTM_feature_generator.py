# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler

window_time = 5
feature_num = 1


def load_data_feature():
    data = pd.read_excel('../data/ERA5.1064.551.xlsx')
    print(data.columns)

    df = data[['speed', 'timestamp']]
    df['speed'] = df.groupby(pd.Grouper(key='timestamp', freq="M"))['speed'].transform('mean')
    df = df.drop_duplicates(['speed'])[['speed', 'timestamp']]
    data_set = df.reset_index()[['speed', 'timestamp']]
    return data_set


def data_generator():
    i = 3
    scaler = MinMaxScaler()
    while i > 0:
        i = i - 1
        from data.utils import read_era5
        data = read_era5(86.94022 + i, 47.55122)
        data['timestamp'] = data.index
        df = data[['speed', 'timestamp']]
        df['speed'] = df.groupby(pd.Grouper(key='timestamp', freq="M"))['speed'].transform('mean')
        data_set = df.drop_duplicates(['speed'])[['speed']].values
        # data_set = df.reset_index()[['speed', 'timestamp']]
        X_org, Y_org = feature_extraction(data_set)
        print("X_org.shape", X_org.shape)
        X = X_org.reshape(-1, X_org.shape[1])
        scaler.fit(X)
        X_scalered = scaler.transform(X)
        X = X_scalered.reshape(X_org.shape[0], X_org.shape[1], feature_num)
        yield X, Y_org


def feature_extraction(data):
    X = []
    Y = []
    for raw in range(len(data) - window_time - 2):
        # for raw in range(300):
        # feature_x = data.loc[raw:raw + window_time, 'speed']
        # feature_y = data.loc[raw + window_time + 1, 'speed']
        # for raw in range(300):
        feature_x = data[raw:raw + window_time]
        feature_y = data[raw + window_time + 1]
        X.append(feature_x)
        Y.append(feature_y)
    return np.array(X), np.array(Y)


def build_model():
    model = Sequential()
    model.add(LSTM(units=10, input_shape=(window_time + 1, feature_num)))
    model.add(Activation('relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model


# 单特征，多特征
# X, Y_org = data_generator()
#
# print(X.shape)
# print(Y_org.shape)

model = build_model()
history = model.fit_generator(data_generator(), epochs=200, verbose=2, steps_per_epoch=10)

plt.plot(history.history['loss'][30:], label='loss')
plt.plot(history.history['val_loss'][30:], label='val_loss')
# plt.plot(history.history['loss'],label='loss')
plt.legend()
plt.show()
