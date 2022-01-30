import pathlib
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

class StockPredictor():
    def plot_predictions(self,test,predicted):
        plt.plot(test[:], color='red',label='Real Stock Close Price')
        plt.plot(predicted[:], color='yellow',label='Predicted Stock Close Price')
        plt.title('Stock Close Price')
        plt.xlabel('Time')
        plt.ylabel('Close Price')
        plt.legend()
        plt.show()

    def __init__(self):
        p= pathlib.Path('~/Documents/TRINIT_OMR_ML04/frontend/deploy/stockpredictor/datasets/daily_IBM.csv')
        self.dataset = pd.read_csv(p)
        self.dataset = self.dataset[['close']]
        self.sc = MinMaxScaler(feature_range=(0,1))
        self.training_set_scaled = self.sc.fit_transform(self.dataset)
        path_name= "~/Documents/TRINIT_OMR_ML04/frontend/deploy/stockpredictor/model/gpvtrvstockrev.h5"
        self.regress = load_model('/home/harry16614/Documents/TRINIT_OMR_ML04/models/ibmstockrev.h5')

    def predict(self):

        X_test = []

        X_test.append(self.training_set_scaled[1:61])
        # y_test.append(testing_set_scaled[i])

        X_test= np.array(X_test)
        X_test = X_test.reshape(1, 60, 1)
        y_pred = self.regress.predict(X_test)
        y_pred = self.sc.inverse_transform(y_pred)
        return y_pred
        # print(y_test, y_pred)
        # self.plot_predictions(y_test, y_pred)
        # X_train = []
        # y_train = []
        # for i in range(60,5520):
        #     X_train.append(training_set_scaled[i-60:i])
        #     y_train.append(training_set_scaled[i])
        # X_train, y_train = np.array(X_train), np.array(y_train)

        # X_train = X_train.reshape(5460, 60, 1)

        # regressor = Sequential()
        # # First LSTM layer with Dropout regularisation
        # regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
        # regressor.add(Dropout(0.2))
        # # Second LSTM layer
        # regressor.add(LSTM(units=50, return_sequences=True))
        # regressor.add(Dropout(0.2))
        # # Third LSTM layer
        # regressor.add(LSTM(units=50, return_sequences=True))
        # regressor.add(Dropout(0.2))
        # # Fourth LSTM layer
        # regressor.add(LSTM(units=50, return_sequences=True))
        # regressor.add(Dropout(0.2))
        # # Fifth LSTM layer
        # regressor.add(LSTM(units=50))
        # regressor.add(Dropout(0.2))
        # # The output layer
        # regressor.add(Dense(units=1))

        # # Compiling the RNN
        # regressor.compile(optimizer='rmsprop',loss='mean_squared_error')
        # # Fitting to the training set
        # regressor.fit(X_train,y_train,epochs=35,batch_size=32)

        # regressor.save("ibmstock.h5")


        