from jqlib.technical_analysis import *
from jqdata import *
import pandas as pd
import numpy as np
import datetime

#from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import tensorflow as tf

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
from numpy import concatenate
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

from keras import backend as K
from keras import optimizers
from keras.layers.core import Dense, Activation, Dropout


def initialize(context):
    # set reference
    set_benchmark('000300.XSHG')
    # set the underlying stock
    g.security = '000001.XSHE'
    g.lookback=10
    start_date = datetime.datetime(2009,1,1)
    end_date = datetime.datetime(2015,1,5)
    stocks = g.security
    df= get_data_from_date(start_date,end_date,stocks)
    result = minmax(df,0.8,g.lookback)
    scaler = result['scaler']
    x_train,x_test,y_train,y_test = result['x_train'],result['x_test'],result['y_train'],result['y_test'] 
    model = lstm_model(x_train,y_train,x_test,y_test)
    g.model = model
    g.scaler = scaler
    
    
    # use real price
    set_option('use_real_price', True)
    
    log.info('start')
    # log.set_level('order', 'error')


    ### 股票相关设定 ###
    # 股票类每笔交易时的手续费是：买入时佣金万分之三，卖出时佣金万分之三加千分之一印花税, 每笔交易佣金最低扣5块钱
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.0003, close_commission=0.0003, min_commission=5), type='stock')

    ## 运行函数（reference_security为运行时间的参考标的；传入的标的只做种类区分，因此传入'000300.XSHG'或'510300.XSHG'是一样的）
      
      # run when market open
    run_daily(market_open, time='close', reference_security='000300.XSHG')
      # run after market close
    run_daily(after_market_close, time='after_close', reference_security='000300.XSHG')


def market_open(context):
    log.info('函数运行时间(market_open):'+str(context.current_dt.time()))
    security = g.security
    # get recent 10 days data
    df_new = get_price(security,end_date=context.current_dt,count=10)
    df_new= df_new[['close','open', 'high', 'low', 'volume', 'money']]
    # get predicted price
    model = g.model
    scaler = g.scaler
    #start_date = datetime.datetime(2009,1,1)
    #end_date = datetime.datetime(2015,1,5)
    #pred_close = lstm(start_date,end_date,10,df_new)
    pred_close = lstm(scaler,model,df_new)
    #get latest close price
    close_new = df_new['close'][-1]
    
    cash = context.portfolio.available_cash

    # buy and sell
    if (close_new > pred_close) and (cash > 0):      
        log.info("price is higher than average 1%%, buy %s" % (security))
        # buy
        order_value(security, cash)
   
    elif close_new < pred_close and context.portfolio.positions[security].closeable_amount > 0:
        log.info("price is lower than average, sell %s" % (security))
        # sell all
        order_target(security, 0)


def after_market_close(context):
    log.info(str('runtime(after_market_close):'+str(context.current_dt.time())))
    
    trades = get_trades()
    for _trade in trades.values():
        log.info('records：'+str(_trade))
    log.info('end day')
    log.info('##############################################################')

##lstm
def lstm(scaler,model,df_new):
    stocks = g.security
    #df= get_data_from_date(start_date,end_date,stocks)
    #result = minmax(df,0.8,lookback)
    #scaler = result['scaler']
    #x_train,x_test,y_train,y_test = result['x_train'],result['x_test'],result['y_train'],result['y_test'] 
    #model = lstm_model(x_train,y_train,x_test,y_test)
    new_data = scaler.transform(df_new)
    new_data = new_data[: g.lookback, 1:]
    new_data = np.reshape(new_data,[1,new_data.shape[0],new_data.shape[1]])
    y_pred = pred(model,new_data,scaler)
    return y_pred
    
    
def get_factors_one_stock(stocks,date):  
    if type(date) != str:
        date = datetime.datetime.strftime(date,'%Y-%m-%d')
    
    price = get_price(stocks,end_date=date,count=1)
    price.index = [date]
    price = price[['close','open', 'high', 'low', 'volume', 'money']]
    return price

def get_data_from_date(start_date,end_date,stocks):
    
    trade_date = get_trade_days(start_date=start_date,end_date=end_date)
    df = get_factors_one_stock(stocks,trade_date[0])
    for date in trade_date[1:]:
        df1 = get_factors_one_stock(stocks,date)
        df = pd.concat([df,df1])
    return df

def minmax(df,train_rate,lookback):
    
    num=int(df.shape[0] * train_rate)
    data_train,data_test=df.iloc[:num,:],df.iloc[num:,:]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data_train)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)
    x_train = np.array([data_train[i : i + lookback, 1:] for i in range(data_train.shape[0] - lookback)])
    y_train = np.array([data_train[i + lookback,0] for i in range(data_train.shape[0]- lookback)])
    x_test = np.array([data_test[i : i + lookback, 1:] for i in range(data_test.shape[0]- lookback)])
    y_test = np.array([data_test[i + lookback,0] for i in range(data_test.shape[0] - lookback)])
    
    np.random.seed(1)
    np.random.shuffle(x_train)
    np.random.seed(1)
    np.random.shuffle(y_train)
    
    len_train = len(x_train)
    len_test = len(x_test)
    n_input = len(df.columns)-1
    x_train = np.reshape(x_train,[len_train,lookback,n_input])
    y_train = np.reshape(y_train,[len_train,1])
    
    x_test = np.reshape(x_test,[len_test,lookback,n_input])
    y_test = np.reshape(y_test,[len_test,1])
    
    result={'scaler':scaler,'x_train':x_train,'y_train':y_train,'x_test':x_test,'y_test':y_test}
    return result

def lstm_model(x_train,y_train,x_test,y_test):
    model = Sequential()
    model.add(LSTM(64, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(loss='mse', optimizer='adam')
    history = model.fit(x_train, y_train, epochs=50, batch_size=100, validation_data=(x_test, y_test), verbose=2,shuffle=False)

    return model


def pred(model,new_data,scaler):
    y_pred = model.predict(new_data)
    inv_y_test = concatenate((y_pred,new_data[:,-1]), axis=1)
    inv_y = scaler.inverse_transform(inv_y_test)

    return inv_y[0][0]
