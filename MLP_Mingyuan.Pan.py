# 导入函数库
# Written by Panming
from jqdata import *
import numpy as np
import keras
from keras.models import load_model
from keras.models import Sequential  # 线性神经网络
from keras.layers.core import Dense, Activation, Dropout  # 神经网络的激活函数
from six import BytesIO


# 初始化函数，设定基准等等
def initialize(context):
    # 设定沪深300作为基准
    set_benchmark('000300.XSHG')
    # 开启动态复权模式(真实价格)
    set_option('use_real_price', True)
    # 输出内容到日志 log.info()
    log.info('The main function will only work once')
    # 过滤掉order系列API产生的比error级别低的log
    # log.set_level('order', 'error')
    body = 'model_MLP'

    ### 股票相关设定 ###
    # 股票类每笔交易时的手续费是：买入时佣金万分之三，卖出时佣金万分之三加千分之一印花税, 每笔交易佣金最低扣5块钱
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.0003, close_commission=0.0003, min_commission=5),
                   type='stock')

    ## 运行函数（reference_security为运行时间的参考标的；传入的标的只做种类区分，因此传入'000300.XSHG'或'510300.XSHG'是一样的）
    # 开盘前运行
    run_daily(before_market_open, time='before_open', reference_security='000300.XSHG')
    # 明天收盘前，预测第二天的涨跌幅进行操作
    run_daily(market_open, time='14:57', reference_security='000300.XSHG')
    # 收盘后运行
    run_daily(after_market_close, time='after_close', reference_security='000300.XSHG')
    g.days = 0  # 记录训练模型间隔，减少内存占用
    g.security = '000001.XSHE'  # 训练的股票
    g.face_back = 10  # 回看天数
    # 查看模型是否存在，不存在，建立开始训练
    # 由于模型是保存在内存中，运行时间会随着周期拉长，大幅拉长，回撤放在半年左右
    try:
        model = load_model('model_MLP')
    except:
        df = get_price(g.security, start_date=None, end_date='2020-1-1', frequency='daily',
                       fields=['open', 'close', 'low', 'high', 'volume', 'money', 'pre_close', ],
                       skip_paused=False, fq='pre', count=1500, panel=True)
        df['rate'] = (df['close'] / df['pre_close'] - 1) * 100
        array = df['rate']
        x = Processing_data(array, g.face_back)
        # y为下一个交易日涨跌幅
        y = array.values[g.face_back:]

        # 建立模型
        def build_model(face_back):
            model = Sequential()
            model.add(Dense(units=12, input_dim=face_back, activation='relu'))
            model.add(Dense(units=6, activation='relu'))
            model.add(Dropout(0.25))
            model.add(Dense(1, activation='linear'))
            model.compile(loss='mse',
                          optimizer='sgd',
                          metrics=['accuracy']
                          )
            return model

        model = build_model(g.face_back)
        model.summary()
        # 模型载入前1500个交易日，训练500次，
        model.fit(x, y, batch_size=20, epochs=500, validation_split=0.2, verbose=0)
        # 保存model
        model.save('model_MLP')


## 开盘前运行函数
# 数据处理函数

def Processing_data(array, face_back=5):
    '''
    把时间系列转换成每列face_back个数的二维数组
    '''
    data = list()
    for i in range(len(array) - face_back):
        a = list(array[i:i + face_back].values)
        data.append(a)
    return np.array(data)


def before_market_open(context):
    # 输出运行时间
    log.info('Function running time (before_market_open)：' + str(context.current_dt.time()))


## 开盘时运行函数
def market_open(context):
    # 无法读取研究中的模型
    # body=read_file('/深度学习/model_MLP')
    df = get_price(g.security, start_date=None, end_date=context.previous_date, frequency='daily',
                   fields=['open', 'close', 'low', 'high', 'volume', 'money', 'pre_close', ],
                   skip_paused=False, fq='pre', count=g.face_back - 1, panel=True)
    df['rate'] = (df['close'] / df['pre_close'] - 1) * 100
    list1 = df['rate'].tolist()

    # data1的数据是当天的数据
    data1 = get_ticks(g.security, end_dt=context.current_dt, start_dt=None, count=3,
                      fields=['time', 'current', 'high', 'low', 'open', 'a1_v'], skip=True, df=False)
    current = (data1['current'][2] / df['close'][-1] - 1) * 100
    list1.append(current)
    a = np.array(list1).reshape(1, g.face_back)
    # 加载模型
    model = load_model('model_MLP')
    # 预测次日涨跌幅
    b = model.predict(a)[0][0]
    log.info("b:" % (b))
    security = g.security
    # 取得当前的现金
    cash = context.portfolio.available_cash
    # 如果第二天预测涨跌幅大于1%，并且有资金，则买入；
    if b > 1 and (cash > 0):
        # 记录这次买入
        log.info("Over rate, buy %s" % (security))
        # 用所有 cash 买入股票
        order_value(security, cash)
    # 如果预计第二天第二大于1%, 则空仓卖出。
    elif b < -1 and context.portfolio.positions[security].closeable_amount > 0:
        # 记录这次卖出
        log.info("Drop over rate, sell %s" % (security))
        # 卖出所有股票,使这只股票的最终持有量为0
        order_target(security, 0)


## 收盘后运行函数
def after_market_close(context):
    log.info(str('Function running time(after_market_close):' + str(context.current_dt.time())))
    # 每g.das盘后加入当天新数据训练十次
    if g.days % 5 == 0:
        model = load_model('model_MLP')
        df = get_price(g.security, start_date=None, end_date=context.current_dt, frequency='daily',
                       fields=['open', 'close', 'low', 'high', 'volume', 'money', 'pre_close', ],
                       skip_paused=False, fq='pre', count=500, panel=True)
        df['rate'] = (df['close'] / df['pre_close'] - 1) * 100
        array = df['rate']
        x = Processing_data(array, g.face_back)
        y = array.values[g.face_back:]
        model.fit(x, y, batch_size=20, epochs=2, validation_split=0.2, verbose=0)
        # 保存model
        model.save('model_MLP')
        g.days = 0
    else:
        g.days += 1
    # #得到当天所有成交记录
    # trades = get_trades()
    # for _trade in trades.values():
    #     log.info('成交记录：'+str(_trade))
    log.info('This day ends')
    log.info('##############################################################')