import warnings
from enum import Enum

import numpy as np
import pandas as pd
from keras import Input
from keras.callbacks import EarlyStopping
from keras.layers import Dense, concatenate
from keras.models import Sequential, Model
from prophet import Prophet
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")
from pandas_datareader import data
from sklearn.preprocessing import StandardScaler
import yfinance as yf
yf.pdr_override()
path = "c:/Windows/Fonts/malgun.ttf"
import platform
from matplotlib import font_manager, rc, pyplot as plt
from abc import abstractmethod, ABCMeta


class ModelType(Enum):
    dnn_model = 1
    dnn_ensemble =2
    lstm_model = 3
    lstm_ensemble =4



class AiTradeBase(metaclass=ABCMeta):
    @abstractmethod
    def split_xy5(self, **kwargs): pass

    @abstractmethod
    def create(self):pass



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

def split_xy5(dataset,time_stpes,y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + int(time_stpes)
        y_end_number = x_end_number + int(y_column)
        if y_end_number > len(dataset):
            break
        temp_x = dataset[i:x_end_number, :]
        temp_y = dataset[x_end_number:y_end_number,3]
        x.append(temp_x)
        y.append(temp_y)
    return np.array(x), np.array(y)


class Kospi(object):
    def __init__(self):
        self.df1 = pd.read_csv(r"C:\Users\AIA\PycharmProjects\djangoProject\admin\dlearn\aitrader\kospi.csv", index_col=0,
                          header=0, encoding='cp949', sep=',')
        self.df2 = pd.read_csv(r"C:\Users\AIA\PycharmProjects\djangoProject\admin\dlearn\aitrader\samsung.csv", index_col=0,
                          header=0, encoding='cp949', sep=',')
        self.samsung = np.load(r"C:\Users\AIA\PycharmProjects\djangoProject\admin\dlearn\aitrader\new_samsung.npy",
                          allow_pickle=True)
        self.kospi200 = np.load(r"C:\Users\AIA\PycharmProjects\djangoProject\admin\dlearn\aitrader\new_kospi.npy",
                           allow_pickle=True)

    def data_load(self):
        df1 = self.df1
        df2 = self.df2
        print(df1)
        print(df1.shape)
        print(df2)
        print(df2.shape)

    def data_preprocessing(self):
        df1 = self.df1
        df2 = self.df2
        for i in range(len(df1.index)):
            df1.iloc[i,4]= int(df1.iloc[i,4].replace(',', ''))
        for i in range(len(df2.index)):
            for j in range(len(df2.iloc[i])):
                df2.iloc[i,j] = int(df2.iloc[i,j].replace(',', ''))
        df1 = df1.sort_values(['일자'],ascending=[True])
        df2 = df2.sort_values(['일자'],ascending=[True])
        print(df1)
        print(df2)
        print(df2.head(6))
        df1 = df1.values
        df2 = df2.values
        print(type(df1), type(df2))
        np.save(r'C:\Users\AIA\PycharmProjects\djangoProject\admin\dlearn\aitrader\new_kospi.npy', arr= df1)
        np.save(r'C:\Users\AIA\PycharmProjects\djangoProject\admin\dlearn\aitrader\new_samsung.npy', arr=df2)
    def create_dnn(self):
        x_test_scaled, x_train_scaled, y_test, y_train = self.basic_scaled()
        model = self.dnn_model(x_train_scaled, y_train)

        loss,mse = model.evaluate(x_test_scaled,y_test, batch_size=1)
        print('loss: ', loss)
        print('mse: ', mse)
        y_prd = model.predict(x_test_scaled)
        for i in range(5):
            print('종가: ',y_test[i], '/ 예측가: ', y_prd[i])
        file_name = r"C:\Users\AIA\PycharmProjects\djangoProject\admin\dlearn\aitrader\save\DNNmodel.h5"
        print(f"저장완료")
        model.save(file_name)

    def dnn_model(self, x_train_scaled, y_train):
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
        return model

    def basic_scaled(self):
        samsung = self.samsung
        samsung = samsung.astype(np.float32)
        x, y = split_xy5(samsung, 5, 1)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, random_state=1, test_size=0.3
        )
        x_train = np.reshape(x_train,
                             (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
        x_test = np.reshape(x_test,
                            (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        return x_test_scaled, x_train_scaled, y_test, y_train

    def create_lstm(self):
        x_test_scaled, x_train_scaled, y_test, y_train = self.basic_scaled()
        x_train_scaled = np.reshape(x_train_scaled,
                                    (x_train_scaled.shape[0],5,5))
        x_test_scaled = np.reshape(x_test_scaled,
                                    (x_test_scaled.shape[0], 5, 5))
        model = self.lstm_model(x_train_scaled, y_train)
        loss,mse = model.evaluate(x_test_scaled,y_test,batch_size=1)
        print('loss: ', loss)
        print('mse : ', mse)
        y_pred = model.predict(x_test_scaled)
        for i in range(5):
            print('종가 : ',y_test[i], '/예측가: ',y_pred[i])

        file_name = r"C:\Users\AIA\PycharmProjects\djangoProject\admin\dlearn\aitrader\save\LSTMmodel.h5"
        print(f"저장완료")
        model.save(file_name)

    def lstm_model(self, x_train_scaled, y_train):
        model = Sequential()
        model.add(Dense(64, input_shape=(5, 5)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        early_stopping = EarlyStopping(patience=20)
        model.fit(x_train_scaled, y_train, validation_split=0.2, verbose=1,
                  batch_size=1, epochs=100, callbacks=[early_stopping])
        return model

    def dnnensemble(self):
        x1_test_scaled, x1_train_scaled, x2_test_scaled, x2_train_scaled, y1_test, y1_train = self.ensemble_scaled()
        model = self.dnn_enselble_model(x1_train_scaled, x2_train_scaled, y1_train)
        loss, mse = model.evaluate([x1_test_scaled, x2_test_scaled],y1_test,
                                   batch_size=1)
        y1_pred = model.predict([x1_test_scaled, x2_test_scaled])
        for i in range(5):
            print('종가 : ', y1_test[i], '/예측가: ', y1_pred[i])

        file_name = r"C:\Users\AIA\PycharmProjects\djangoProject\admin\dlearn\aitrader\save\DNNensemblemodel.h5"
        print(f"저장완료")
        model.save(file_name)

    def dnn_enselble_model(self, x1_train_scaled, x2_train_scaled, y1_train):
        input1 = Input(shape=(25,))
        dense1 = Dense(64)(input1)
        dense1 = Dense(32)(dense1)
        dense1 = Dense(32)(dense1)
        output1 = Dense(32)(dense1)
        input2 = Input(shape=(25,))
        dense2 = Dense(64)(input2)
        dense2 = Dense(64)(dense2)
        dense2 = Dense(64)(dense2)
        dense2 = Dense(64)(dense2)
        output2 = Dense(32)(dense2)
        merge = concatenate([output1, output2])
        output3 = Dense(1)(merge)
        model = Model(inputs=[input1, input2],
                      outputs=output3)
        model.compile(loss='mse', optimizer='adam',
                      metrics=['mse'])
        early_stopping = EarlyStopping(patience=20)
        model.fit([x1_train_scaled, x2_train_scaled], y1_train, validation_split=0.2,
                  verbose=1, batch_size=1, epochs=100,
                  callbacks=[early_stopping])
        return model

    def ensemble_scaled(self):
        samsung = self.samsung
        kospi200 = self.kospi200
        samsung = samsung.astype(np.float32)
        kospi200 = kospi200.astype(np.float32)
        x1, y1 = split_xy5(samsung, 5, 1)
        x2, y2 = split_xy5(kospi200, 5, 1)
        x1_train, x1_test, y1_train, y1_test = train_test_split(
            x1, y1, random_state=1, test_size=0.3)
        x2_train, x2_test, y2_train, y2_test = train_test_split(
            x2, y2, random_state=2, test_size=0.3)
        x1_train = np.reshape(x1_train,
                              (x1_train.shape[0], x1_train.shape[1] * x1_train.shape[2]))
        x1_test = np.reshape(x1_test,
                             (x1_test.shape[0], x1_test.shape[1] * x1_test.shape[2]))
        x2_train = np.reshape(x2_train,
                              (x2_train.shape[0], x2_train.shape[1] * x2_train.shape[2]))
        x2_test = np.reshape(x2_test,
                             (x2_test.shape[0], x2_test.shape[1] * x2_test.shape[2]))
        scaler1 = StandardScaler()
        scaler1.fit(x1_train)
        x1_train_scaled = scaler1.transform(x1_train)
        x1_test_scaled = scaler1.transform(x1_test)
        scaler2 = StandardScaler()
        scaler2.fit(x2_train)
        x2_train_scaled = scaler2.transform(x2_train)
        x2_test_scaled = scaler2.transform(x2_test)
        return x1_test_scaled, x1_train_scaled, x2_test_scaled, x2_train_scaled, y1_test, y1_train

    def lstmensemble(self):
        x1_test_scaled, x1_train_scaled, x2_test_scaled, x2_train_scaled, y1_test, y1_train = self.ensemble_scaled()
        x1_train_scaled = np.reshape(x1_train_scaled,
                                    (x1_train_scaled.shape[0],5,5))
        x1_test_scaled = np.reshape(x1_test_scaled,
                                    (x1_test_scaled.shape[0], 5, 5))

        x2_train_scaled = np.reshape(x2_train_scaled,
                                    (x2_train_scaled.shape[0],5,5))
        x2_test_scaled = np.reshape(x2_test_scaled,
                                    (x2_test_scaled.shape[0], 5, 5))
        model = self.lstm_ensemble_model(x1_train_scaled, x2_train_scaled, y1_train)
        loss, mse = model.evaluate([x1_test_scaled, x2_test_scaled], y1_test,
                                   batch_size=1)
        y1_pred = model.predict([x1_test_scaled, x2_test_scaled])
        for i in range(5):
            print('종가 : ', y1_test[i], '/예측가: ', y1_pred[i])

        file_name = r"C:\Users\AIA\PycharmProjects\djangoProject\admin\dlearn\aitrader\save\LSTMensemblemodel.h5"
        print(f"저장완료")
        model.save(file_name)

    def lstm_ensemble_model(self, x1_train_scaled, x2_train_scaled, y1_train):
        input1 = Input(shape=(5, 5))
        dense1 = Dense(64)(input1)
        dense1 = Dense(32)(dense1)
        dense1 = Dense(32)(dense1)
        output1 = Dense(32)(dense1)
        input2 = Input(shape=(5, 5))
        dense2 = Dense(64)(input2)
        dense2 = Dense(64)(dense2)
        dense2 = Dense(64)(dense2)
        dense2 = Dense(64)(dense2)
        output2 = Dense(32)(dense2)
        merge = concatenate([output1, output2])
        output3 = Dense(1)(merge)
        model = Model(inputs=[input1, input2],
                      outputs=output3)
        model.compile(loss='mse', optimizer='adam',
                      metrics=['mse'])
        early_stopping = EarlyStopping(patience=20)
        model.fit([x1_train_scaled, x2_train_scaled], y1_train, validation_split=0.2,
                  verbose=1, batch_size=1, epochs=100,
                  callbacks=[early_stopping])
        return model
