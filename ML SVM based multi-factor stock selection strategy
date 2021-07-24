# This strategy is to be used on joinquant

from jqdata import *
import jqdata
import pandas as pd
import numpy as np
import datetime
from jqfactor import standardlize
from jqfactor import winsorize_med
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression


def initialize(context):
    pd.set_option('display.max_columns', None)
    
    pd.set_option('display.max_rows', None)
    
    # set 000300.XSHG as the reference
    set_benchmark('000300.XSHG')

    set_option('use_real_price', True)

    log.info('begin')


    
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.0003, close_commission=0.0003, min_commission=5), type='stock')
    run_monthly(func,monthday=1,time='open', reference_security='000300.XSHG')
    
    
    
def func(context):
    g.industrials = ['801110','801120', '801750']#申万一级行业

    g.code = '000300.XSHG'
    stock_code = get_index_stocks(g.code)
 
    g._stocks = set_feasible_stocks(context,stock_code)

    g.tradingday = tradingday(context)
    #print(g.tradingday)
   
    g._q = get_q_factor(context,g._stocks)
 
    df = get_fundamentals(g._q)
    #print(df)
    d1 = get_fundamentals(g._q,g.tradingday[-2])
    #print(d1)
    d2 = get_fundamentals(g._q,g.tradingday[-3])
    d3 = get_fundamentals(g._q,g.tradingday[-4])
    df_train = pd.concat([d1, d2, d3],ignore_index=True)
    #print(df_train.columns)
    
    # process the data
    df = data_factor(context,df)
    #print(df)
    df =df.set_index(['code'])
    df.datas = df.dropna(axis=0)
    #print(df)
    df_train = data_factor(context,df_train)
    df_train = df_train.set_index(['code'])
    df_train = df_train.dropna(axis=0)

    df_columns = df.columns
    df_train_columns = df_train.columns
    
    for fac in df_columns:
        df_train[fac] = winsorize_med(df_train[fac], scale=5, inclusive=True, inf2nan=True, axis=0)    
        df[fac] = winsorize_med(df[fac], scale=5, inclusive=True, inf2nan=True, axis=0)   
    # standard
    for fac in df_train_columns:
        df_train[fac] = standardlize(df_train[fac], inf2nan=True, axis=0)
        df[fac] = standardlize(df[fac], inf2nan=True, axis=0)
    # neutralize
    g.__industry_set = get_industries('jq_l1', date=context.current_dt)
    #print(g.__industry_set)
    g.__industry_set=g.__industry_set.index
    df = neutralize(df,g.__industry_set)
    df_train = neutralize(df_train,g.__industry_set)

    # training set
    df_columns_factor = df.columns[1:]
    df_train_columns_factor = df_train.columns[1:]
    X_trainval = df_train[df_train_columns_factor] 
    X_trainval = X_trainval.fillna(0)
    y_trainval = df_train[['market_cap']]
    y_trainval = y_trainval.fillna(0)
    
    # testing set
    X = df[df_columns_factor]
    X = X.fillna(0)
    y = df[['market_cap']]
    y = y.fillna(0)   
    svm = SVR(C=100, gamma=1)    

 # fit the model
    svm.fit(X_trainval,y_trainval)
    # predict market_cap
    y_pred = svm.predict(X)
    # 新的因子：实际值与预测值之差    
    g.factor1 = y - pd.DataFrame(y_pred, index = y.index, columns = ['market_cap'])
   
    stockset=select_stock(context)
    
    # prepare data
    current_data = get_current_data()
    hs_price=history(50, unit='1d', field='close', security_list='000300.XSHG', df=True, skip_paused=False, fq='pre')['000300.XSHG']
    sell_list = list(context.portfolio.positions.keys())
    # sell
    for stock in sell_list:
        if stock not in stockset:
            if stock in g._stocks :
                if current_data[stock].last_price == current_data[stock].high_limit:
                    pass
                else:
                    stock_sell = stock
                    order_target_value(stock_sell, 0)    
    
  
    # allocate portfolio
    s_sum=len(set(stockset)-set(context.portfolio.positions))
    position = context.portfolio.total_value/s_sum
    # buy
    if hs_price[-1]>hs_price.mean():
        for stock in stockset:
            if stock in sell_list:
                pass
            else:
                if current_data[stock].last_price == current_data[stock].low_limit:
                    pass
                else:
                    order_value(stock, position)     
    
    
    

def select_stock(context):
    stocks = []
    g._stock = g.factor1.index
    for ix in g.industrials:
        istocks = get_industry_stocks(ix)
        g.stock_list1 = [s for s in istocks if s in g._stock]
        q=query(valuation.code).filter(valuation.pb_ratio > 0,valuation.pe_ratio > 0,valuation.pcf_ratio > 0,valuation.code.in_(g.stock_list1)).order_by((valuation.market_cap/valuation.pcf_ratio).desc())
        df = get_fundamentals(q).dropna()
        stocks = stocks + list(df['code'])
        stock_list=pd.DataFrame() 
        for i in stocks:
            g1=pd.DataFrame(g.factor1.loc[i,:]).T
            stock_list=pd.concat([stock_list,g1])
    #print(stock_list)
    stock_list=stock_list.sort_values(by = 'market_cap',ascending=False)
    g.stocknum = int(len(stock_list)/5)
    stock_list3=stock_list.index[0:g.stocknum]
    return stock_list3


    
# neutralize data
def neutralize(df,industry_set):
    for i in range(len(industry_set)):
        s = pd.Series([0]*len(df), index=df.index)
        df[industry_set[i]] = s

        industry = get_industry_stocks(industry_set[i])
        for j in range(len(df)):
            if df.iloc[j,0] in industry:
                df.iloc[j,i+8] = 1
                
    return df 

 
# remove extreme values
def mad(factor):
    me = np.median(factor)
    mad = np.median(abs(factor-me))
    up = me+(3* 1.4826*mad)
    down =me-(3* 1.4826*mad)
    factor = np.where(factor>up,up,factor)
    factor = np.where(factor<down,down,factor)
    return factor


# get standard values
def stand(factor):
    mean = factor.mean()
    std = factor.std()
    return (factor-mean)/std
 
 
    
# data factor processing
def data_factor(context,df_):
    df_['market_cap'] = np.log(df_['market_cap'])
    df_['pe_ratio'] = df_['pe_ratio'].apply(lambda x: 1/x)
    df_['pb_ratio'] = df_['pb_ratio'].apply(lambda x: 1/x)
    df_['pcf_ratio'] = df_['pcf_ratio'].apply(lambda x: 1/x)
    df_['anon_1'] = np.log(df_['anon_1'])#净资产
    df_['circulating_market_cap'] = np.log(df_['circulating_market_cap'])
    df_['net_profit_to_total_revenue'] = np.abs(df_['net_profit_to_total_revenue'])
    df_['peg'] = df_['pe_ratio']/(df_['inc_net_profit_year_on_year']*100)
    #print(df_)
    return df_

    

# get data factors query
def get_q_factor(context,feasible_stocks):
    q = query(valuation.code, 
          valuation.market_cap,#市值
          valuation.circulating_market_cap,#流通市值
          valuation.turnover_ratio,#换手率
          valuation.pe_ratio, #市盈率（TTM）
          valuation.ps_ratio, #市销率
          valuation.pb_ratio, #市净率（TTM）
          valuation.pcf_ratio, #市现率
          balance.total_assets - balance.total_liability,#净资产
          balance.total_liability / balance.total_assets, #资产负债率=负债/资产
          balance.fixed_assets / balance.total_assets, #固定资产占比
          indicator.net_profit_to_total_revenue, #净利润/营业总收入
          indicator.inc_revenue_year_on_year,  #营业收入增长率（同比）
          indicator.inc_net_profit_year_on_year,#净利润增长率（同比）
          indicator.roe,#净资产收益率
          indicator.roa,#资产收益率
          indicator.gross_profit_margin, #销售毛利率GPM
        ).filter(
            valuation.code.in_(feasible_stocks)
        )
    return q


    
# set the trading market
def set_feasible_stocks(context,initial_stocks):
    paused_info = []
    current_data = get_current_data()
    for i in initial_stocks:
        paused_info.append(current_data[i].paused)
    df_paused_info = pd.DataFrame({'paused_info':paused_info},index = initial_stocks)
    unsuspened_stocks =list(df_paused_info.index[df_paused_info.paused_info == False])
    return unsuspened_stocks   

#get the data of the previous 4 months
def tradingday(context):
    g.trading_days = get_all_trade_days()
    g.date_=context.current_dt
    g.trading_days = [x for x in g.trading_days if x <= datetime.date(g.date_.year,g.date_.month,g.date_.day)]
    g.tradingday = []
    for i in range(len(g.trading_days)-1):
        if g.trading_days[i].year != g.trading_days[i+1].year:
            g.tradingday.append(g.trading_days[i+1])
        elif g.trading_days[i].month != g.trading_days[i+1].month:
            g.tradingday.append(g.trading_days[i+1])
    g.tradingday = g.tradingday[-5:-1] 
    date_now=context.current_dt
    date_now=datetime.date(date_now.year,date_now.month,date_now.day)
    #print(date_now)
    date1=[]
    for i in g.tradingday:
        date1.append(i)
    #g.tradingday=g.tradingday.append(datetime.date(date_now.year,date_now.month,date_now.day))
    date1.append(date_now)
    g.tradingday1=date1
    return g.tradingday1 
