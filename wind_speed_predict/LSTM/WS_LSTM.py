
# coding: utf-8

# In[108]:


import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# In[109]:


window_time = 5
data = pd.read_excel('ERA5.1064.551.xlsx')
df = data[['timestamp', 'speed']]
df['speed'] = df.groupby(pd.Grouper(key='timestamp', freq="M"))['speed'].transform('mean')
df = df.drop_duplicates(['speed'])[['timestamp', 'speed']]
data=df.reset_index()[['timestamp', 'speed']]

def load_data_month():  
    X = []
    Y = []
    for raw in range(len(data) - window_time - 2):
        # for raw in range(300):
        feature_x = data.loc[raw:raw + window_time, ['speed']].values
        feature_y = data.loc[raw + window_time + 1, 'speed']
        X.append(np.transpose(feature_x))
        Y.append(np.transpose(feature_y))
    return np.array(X), np.array(Y)


# In[113]:


X, Y = load_data_month()

# 归一化
X = X.reshape(X.shape[0],X.shape[2])
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
X = X.reshape(X.shape[0],1,X.shape[1])
# 划分测试集，验证集
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.33, random_state=42)
X_train[0]
X_train.shape


# In[125]:


from sklearn.model_selection import train_test_split
from keras.layers import Activation, Dense
X, Y = load_data_month()
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.33, random_state=42)
X_train.shape
model = Sequential()
model.add(LSTM(units=10, input_shape=(X.shape[1], X.shape[2])))
# model.add(Dense(10, activation='softmax'))
model.add(Activation('relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
# model.add(Activation('relu'))
# compile:loss, optimizer, metrics
model.compile(loss='mse', optimizer='adam')
model.summary()
# train: epcoch, batch_size
# history = model.fit(X_train, y_train, epochs=4000, batch_size=5, verbose=0)
history = model.fit(X_train, y_train, epochs=700, batch_size=5, verbose=2,validation_data=(X_test,y_test))


# In[123]:


plt.plot(history.history['loss'][30:],label='loss')
plt.plot(history.history['val_loss'][30:],label='val_loss')
# plt.plot(history.history['loss'],label='loss')
plt.legend()
plt.show()


# In[124]:


model.summary()
# score = model.evaluate(X_test, y_test, batch_size=15, verbose=1)
# print(score)
y_pre = model.predict(X_test)

import matplotlib.pyplot as plt
# plt.plot(history.history['mse'])
# plt.show()

plt.plot(y_test, label='y_test')
plt.plot(y_pre, label='y_pre')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




