from turtle import st
import streamlit as st
from keras.saving.save import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

st.title("Stock Prediction")
stocks = ("IBM", "DaiDex", "GPVTRV", "Reliance", "ShopTRT", "SAIC", "TSCOLON", "VANKE")
selected_stocks = st.selectbox("Select stock fpr prediction", stocks)

if(selected_stocks=="IBM"):
    model=load_model("ibmstockrev.h5")
    dataset = pd.read_csv('daily_IBM.csv')
if(selected_stocks=="DaiDex"):
    model=load_model("daidexstockrev.h5")
    dataset = pd.read_csv('daily_DAIDEX.csv')
if(selected_stocks=="GPVTRV"):
    model=load_model("gpvtrvstockrev.h5")
    dataset = pd.read_csv('daily_GPVTRV.csv')
if(selected_stocks=="Reliance"):
    model=load_model("reliancestockrev.h5")
    dataset = pd.read_csv('daily_RELIANCEBSE.csv')
if(selected_stocks=="ShopTRT"):
    model=load_model("shopstrtstockrev.h5")
    dataset = pd.read_csv('daily_SHOPTRT.csv')
if(selected_stocks=="SAIC"):
    model=load_model("saicshhstockrev.h5")
    dataset = pd.read_csv('daily_SIACSHH.csv')
if(selected_stocks=="TSCOLON"):
    model=load_model("tscolonstockrev.h5")
    dataset = pd.read_csv('daily_TSCOLON.csv')
if(selected_stocks=="VANKE"):
    model=load_model("vankestockrev.h5")
    dataset = pd.read_csv('daily_VANKE.csv')

dataset = dataset[['close']]
sc = MinMaxScaler(feature_range=(0,1))
testing_set_scaled = sc.fit_transform(dataset)
X_test = []
y_test = []

X_test.append(testing_set_scaled[0:60])
y_test.append(testing_set_scaled[0])

X_test = np.array(X_test)
X_test = X_test.reshape(1, 60, 1)
y_pred = model.predict(X_test)
y_pred = sc.inverse_transform(y_pred)

st.subheader('Forecasted Close Value')
st.write(y_pred)

st.subheader('Previous Day Close Value')    
st.write(sc.inverse_transform(y_test))