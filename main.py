import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
plt.style.use("fivethirtyeight")
import plotly.graph_objs as go
import webbrowser

from pandas_datareader.data import DataReader
import yfinance as yf

from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import LSTM, Dense
from streamlit_lottie import st_lottie
import requests

from PIL import Image

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()



def main():
    st.sidebar.title('Stock Market Prediction 📊')
    with st.sidebar:
        lottie_sidebar = 'https://assets1.lottiefiles.com/private_files/lf30_fgba6oco.json'
        lottie_json = load_lottieurl(lottie_sidebar)
        st_lottie(lottie_json, key='lottie_sidebar')

    menu = ["Introduction","Home","Visualizations","Prediction","About"]
    choice = st.sidebar.selectbox("Menu 📋", menu)

    end = st.sidebar.date_input('End Date 📆', datetime.now())
    start = st.sidebar.date_input('Start Date 📆', datetime(end.year - 3, end.month, end.day))
    stock_ticker = st.sidebar.text_input('Stock ticker in CAPS',placeholder='Enter stock ticker',value='MSFT')
    df = DataReader(stock_ticker, data_source='yahoo', start=start, end=end)


    if choice == "Introduction":
        st.title('Stock Market Analysis 📈 ')

        intro = yf.Ticker(stock_ticker)
        #st.write(intro.info)

        col1, col2,col3 = st.columns(3)

        with col1:
            logo = intro.info['logo_url']
            st.image(logo)
        with col2:
            comp_name = st.subheader(intro.info['longName'])
            comp_country = intro.info['country']
            comp_city = intro.info['city']
            st.write(comp_country)
            st.write(comp_city)
        with col3:
            lottie_intro = 'https://assets3.lottiefiles.com/packages/lf20_wh4gk3bb.json'
            lottie_json = load_lottieurl(lottie_intro)
            st_lottie(lottie_json, key='lottie_intro')

        with st.expander('Company Description'):
            st.subheader('Description')
            st.write(intro.info['longBusinessSummary'])
        comp_url = intro.info['website']
        if st.button('Visit Website'):
            webbrowser.open_new_tab(comp_url)

    if choice == "Home":

        col8,col9 = st.columns(2)
        with col8:
            st.title('Going through the Dataset 📝 and Getting an Overview ...')

        with col9:
            lottie_home = 'https://assets4.lottiefiles.com/private_files/lf30_t8g5t1is.json'
            lottie_json = load_lottieurl(lottie_home)
            st_lottie(lottie_json, key='lottie_home')

        st.subheader('Last 10 Days overview')
        st.table(df.tail(10).reset_index())

        st.subheader('Insights of the Data')
        st.write(df.describe())

    if choice == "Visualizations":

        st.title('Visualization of Stock Data 📊')
        col4,col5 = st.columns(2)
        with col4:
            st.subheader('Open price Vs Time Chart')
            fig = plt.figure(figsize=(12, 6))
            plt.xlabel('Years')
            plt.ylabel('Open Price')
            plt.plot(df['Open'],color='green')
            st.pyplot(fig)

        with col5:
            st.subheader('Closing price Vs Time Chart')
            fig = plt.figure(figsize=(12, 6))
            plt.xlabel('Years')
            plt.ylabel('Close Price')
            plt.plot(df.Close)
            st.pyplot(fig)

        col12,col13 = st.columns(2)
        with col12:
            st.subheader('Open price Vs Close Chart')
            fig = plt.figure(figsize=(12, 6))
            plt.xlabel('Years')
            plt.ylabel('Close Price & Open Price')
            plt.plot(df.Close)
            plt.plot(df.Open)
            st.pyplot(fig)

        with col13:
            st.subheader('High Vs Low')
            fig = plt.figure(figsize=(12, 6))
            plt.xlabel('Years')
            plt.ylabel('High & Low')
            plt.plot(df.High)
            plt.plot(df.Low)
            st.pyplot(fig)

        col6,col7 = st.columns(2)

        with col6:
            st.subheader('CLOSE - 100 Days Moving Average')
            fig = plt.figure(figsize=(12, 6))
            ma100 = df.Close.rolling(100).mean()
            plt.xlabel('Years')
            plt.ylabel('Close Price')
            plt.plot(df.Close)
            plt.plot(ma100)
            st.pyplot(fig)
        with col7:
            st.subheader('CLOSE - 100 Vs 200 Days Moving Average')
            fig = plt.figure(figsize=(12, 6))
            ma100 = df.Close.rolling(100).mean()
            ma200 = df.Close.rolling(200).mean()
            plt.xlabel('Years')
            plt.ylabel('Close Price')
            plt.plot(df.Close)
            plt.plot(ma100)
            plt.plot(ma200)
            st.pyplot(fig)

    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = int(np.ceil(len(dataset) * .95))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)


    train_data = scaled_data[0:int(training_data_len), :]
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

    if choice == "Prediction":
        st.title('Final Prediction 📉')

        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid['Predictions'] = predictions
        # Visualize the data
        fig = plt.figure(figsize=(16, 6))
        plt.title('Model')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'Predictions']])
        plt.legend(['Train', 'Valid', 'Predictions'], loc='upper left')
        st.pyplot(fig)

        col10,col11 = st.columns(2)
        with col10:
            #calculate = y_test/predictions *100
            #sum = 0
            #for i in calculate:
             #   sum=sum+i
            #accuracy = sum / len(y_test)

            st.subheader('Original Vs Predicted')
            st.write(valid.reset_index())

            #st.subheader('Accuracy of the Model : {} % '.format(accuracy))
        with col11:
            lottie_final = 'https://assets1.lottiefiles.com/packages/lf20_ruflv73p.json'
            lottie_json = load_lottieurl(lottie_final)
            st_lottie(lottie_json, key='lottie_final')

    if choice =='About':
        st.title('About')
        image = Image.open('aboutus.png')
        st.image(image)


if __name__ == "__main__":
    main()
