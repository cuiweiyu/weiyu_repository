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


def load_data_feature():
    data = pd.read_excel('../data/ERA5.1064.551.xlsx')
    print(data.columns)

    df = data[['speed', 'timestamp']]
    df['speed'] = df.groupby(pd.Grouper(key='timestamp', freq="M"))['speed'].transform('mean')
    df = df.drop_duplicates(['speed'])[['speed', 'timestamp']]
    data_set = df.reset_index()[['speed', 'timestamp']]
    return data_set


def feature_extraction(data):
    X = []
    Y = []
    for raw in range(len(data) - window_time - 2):
        # for raw in range(300):
        feature_x = data.loc[raw:raw + window_time, 'speed']
        feature_y = data.loc[raw + window_time + 1, 'speed']
        X.append(feature_x)
        Y.append(feature_y)
    return np.array(X), np.array(Y)


def build_model():
    model = Sequential()
    model.add(LSTM(units=10, input_shape=(X.shape[1], X.shape[2])))
    model.add(Activation('relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model


# 单特征，多特征
data = load_data_feature()
X_org, Y_org = feature_extraction(data)
# data = load_data_features()
# X_org, Y_org = features_extraction()
print(X_org.shape)
print(Y_org.shape)
# print(X_org[:3])


# 归一化
# X = X_org.reshape(-1, X_org.shape[1])
X = X_org
scaler = MinMaxScaler()
scaler.fit(X)
X_scalered = scaler.transform(X)
X = X_scalered.reshape(X_org.shape[0], X_org.shape[1], 1)
# print(X[:3])
print(X.shape)
print(Y_org.shape)

model = build_model()
history = model.fit(X, Y_org, epochs=200, batch_size=5, verbose=2, validation_split=0.2)

plt.plot(history.history['loss'][30:], label='loss')
plt.plot(history.history['val_loss'][30:], label='val_loss')
# plt.plot(history.history['loss'],label='loss')
plt.legend()
plt.show()
