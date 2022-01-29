from keras.saving.save import load_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


def plot_predictions(test,predicted):
    plt.plot(test[:], color='red',label='Real Stock Close Price')
    plt.plot(predicted[:], color='yellow',label='Predicted Stock Close Price')
    plt.title('Stock Close Price')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()


dataset = pd.read_csv('daily_IBM.csv')
dataset = dataset[['close']]
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(dataset)
print(training_set_scaled)
print(dataset)

X_train = []
y_train = []
for i in range(60,5520):
    X_train.append(training_set_scaled[i-60:i])
    y_train.append(training_set_scaled[i])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = X_train.reshape(5460, 60, 1)
print(X_train, X_train.shape)

regressor = Sequential()
# First LSTM layer with Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))
# Second LSTM layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
# Third LSTM layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
# Fourth LSTM layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
# Fifth LSTM layer
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
# The output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='rmsprop',loss='mean_squared_error')
# Fitting to the training set
regressor.fit(X_train,y_train,epochs=35,batch_size=32)

regressor.save("ibmstock.h5")

# regress = load_model("ibmstock.h5")

dataset = pd.read_csv('daily_IBM.csv')
dataset = dataset[['close']]
sc = MinMaxScaler(feature_range=(0,1))
testing_set_scaled = sc.fit_transform(dataset)
X_test = []
y_test = []

i=100
X_test.append(testing_set_scaled[i-60:i])
y_test.append(testing_set_scaled[i])

X_test, y_test = np.array(X_test), np.array(y_test)
X_test = X_test.reshape(1, 60, 1)
y_pred = regressor.predict(X_test)
print(y_test, y_pred)
plot_predictions(y_test, y_pred)