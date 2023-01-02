import warnings

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
from prophet import Prophet
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")
import pandas_datareader.data as web
from pandas_datareader import data
from sklearn.preprocessing import StandardScaler
import yfinance as yf
yf.pdr_override()
path = "c:/Windows/Fonts/malgun.ttf"
import platform
from matplotlib import font_manager, rc, pyplot as plt

if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
else:
    print('Unknown system... sorry~~~~')
plt.rcParams['axes.unicode_minus'] = False


'''
Date  Open     High      Low    Close     Adj Close   Volume
'''




class AiTraderService(object):
    def __init__(self):
        global start_date, end_date, item_code
        start_date = "2018-1-4"
        end_date = "2021-9-30"
        item_code = "000270.KS"

    def hook(self):
        item = data.get_data_yahoo(item_code, start_date, end_date)
        print(f" KIA head: {item.head(3)}")
        print(f" KIA tail: {item.tail(3)}")
        item['Close'].plot(figsize=(12, 6), grid=True)
        item_trunc = item[:'2021-12-31']
        df = pd.DataFrame({'ds': item_trunc.index, 'y': item_trunc['Close']})
        df.reset_index(inplace=True)
        del df['Date']
        prophet = Prophet(daily_seasonality=True)
        prophet.fit(df)
        future = prophet.make_future_dataframe(periods=61)
        forecast = prophet.predict(future)
        prophet.plot(forecast)
        plt.figure(figsize=(12, 6))
        plt.plot(item.index, item['Close'], label='real')
        plt.plot(forecast['ds'], forecast['yhat'], label='forecast')
        plt.grid()
        plt.legend()
        path = r"C:\Users\AIA\PycharmProjects\djangoProject\admin\dlearn\aitrader"
        print(f"path:{path}")
        plt.savefig(f'{path}\\kia_close.png')


class Kospi(object):
    def __init__(self):
        global df1, df2
        df1 = pd.read_csv(r"C:\Users\AIA\PycharmProjects\djangoProject\admin\dlearn\aitrader\kospi.csv", index_col=0,
                          header=0, encoding='cp949', sep=',')
        df2 = pd.read_csv(r"C:\Users\AIA\PycharmProjects\djangoProject\admin\dlearn\aitrader\samsung.csv", index_col=0,
                          header=0, encoding='cp949', sep=',')

    def data_load(self):
        print(df1)
        print(df1.shape)
        print(df2)
        print(df2.shape)

    def data_preprocessing(self):
        df1 = pd.read_csv(r"C:\Users\AIA\PycharmProjects\djangoProject\admin\dlearn\aitrader\kospi.csv", index_col=0,
                          header=0, encoding='cp949', sep=',')
        df2 = pd.read_csv(r"C:\Users\AIA\PycharmProjects\djangoProject\admin\dlearn\aitrader\samsung.csv", index_col=0,
                          header=0, encoding='cp949', sep=',')
        for i in range(len(df1.index)):
            df1.iloc[i,4]= int(df1.iloc[i,4].replace(',', ''))
        for i in range(len(df2.index)):
            for j in range(len(df2.iloc[i])):
                df2.iloc[i,j] = int(df2.iloc[i,j].replace(',', ''))
        df1 = df1.sort_values(['일자'],ascending=[True])
        df2 = df2.sort_values(['일자'],ascending=[True])
        print(df1)
        print(df2)
        df1 = df1.values
        df2 = df2.values
        print(type(df1), type(df2))
        print(df1.shape, df2.shape)
        np.save(r'C:\Users\AIA\PycharmProjects\djangoProject\admin\dlearn\aitrader\new_kospi.npy', arr= df1)
        np.save(r'C:\Users\AIA\PycharmProjects\djangoProject\admin\dlearn\aitrader\new_samsung.npy', arr=df2)

    def data_reloading(self):
        kospi200 = np.load(r"C:\Users\AIA\PycharmProjects\djangoProject\admin\dlearn\aitrader\new_kospi.npy",allow_pickle=True)
        samsung = np.load(r"C:\Users\AIA\PycharmProjects\djangoProject\admin\dlearn\aitrader\new_samsung.npy",allow_pickle=True)
        print(kospi200)
        print(samsung)
        print(kospi200.shape)
        print(samsung.shape)



    def splitsamsung(self):
        samsung = r"C:\Users\AIA\PycharmProjects\djangoProject\admin\dlearn\aitrader\new_samsung.npy"
        x, y = split_xy5(samsung, 5, 1)
        x_train, x_test, y_train, y_test = train_test_split(
            x,y,random_state=1,test_size=0.3
        )
        x_train = np.reshape(x_train,
                             (x_train.shape[0], x_train.shape[1 ] * x_train.shape[2]))
        x_test = np.reshape(x_test,
                             (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        model = Sequential()
        model.add(Dense(64, input_shape=(25,)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        early_stopping = EarlyStopping(patience=20)
        model.fit(x_train_scaled, y_train, validation_split=0.2, verbose=1,
                  batch_size=1, epochs=100, callbacks=[early_stopping])

        loss,mse = model.evaluate(x_test_scaled,y_test, batch_size=1)
        print('loss: ', loss)
        print('mse: ', mse)
        y_prd = model.predict(x_test_scaled)
        for i in range(5):
            print('종가: ',y_test[i], '/ 예측가: ', y_prd[i])

def split_xy5(dataset,time_stpes,y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + int(time_stpes)
        y_end_number = x_end_number + int(y_column)
        if y_end_number > len(dataset):
            break
        temp_x = dataset[i:x_end_number, :]
        temp_y = dataset[x_end_number:y_end_number,3 ]
        x.append(temp_x)
        y.append(temp_y)
    return np.array(x), np.array(y)

if __name__ == '__main__':
    ai = AiTraderService()
    k = Kospi()
    k.splitsamsung()