# -*- coding: utf-8 -*-
import pandas as pd
import os
import sys
import numpy as np

import matplotlib

matplotlib.use('Agg')
from matplotlib.pyplot import plot, savefig
from matplotlib import pyplot as plt

pd.set_option('display.width', 800)

datapath = './daily'
filepath = os.path.join(datapath, os.listdir('./daily')[0])

import re

ticker_regex = re.compile('.+_(?P<ticker>.+)\.csv')
get_ticker = lambda x: ticker_regex.match(x).groupdict()['ticker']
print(filepath, get_ticker(filepath))

ret = lambda x, y: np.log(y / x)  # Log return
zscore = lambda x: (x - x.mean()) / x.std()  # zscore

D = pd.read_csv(filepath, header=None, names=['UNK', 'o', 'h', 'l', 'c', 'v'])  # Load the dataframe with headers
print D.head(10)
print "-" * 20, "Example csv data", "-" * 20


# 各指标都是和open价格做比值，在标准化分
def make_inputs(filepath):
    D = pd.read_csv(filepath, header=None, names=['UNK', 'o', 'h', 'l', 'c', 'v'])  # Load the dataframe with headers
    D.index = pd.to_datetime(D.index, format='%Y%m%d')  # Set the indix to a datetime
    Res = pd.DataFrame()
    ticker = get_ticker(filepath)

    Res['c_2_o'] = zscore(ret(D.o, D.c))
    Res['h_2_o'] = zscore(ret(D.o, D.h))
    Res['l_2_o'] = zscore(ret(D.o, D.l))
    Res['c_2_h'] = zscore(ret(D.h, D.c))
    Res['h_2_l'] = zscore(ret(D.h, D.l))
    Res['c1_c0'] = ret(D.c, D.c.shift(-1)).fillna(0)  # Tommorows return
    Res['vol'] = zscore(D.v)
    Res['ticker'] = ticker
    return Res


Res = make_inputs(filepath)
# print Res.head()
# print Res.corr()

Final = pd.DataFrame()
idx = 0
for f in os.listdir(datapath):
    filepath = os.path.join(datapath, f)
    if filepath.endswith('.csv'):
        Res = make_inputs(filepath)
        Final = Final.append(Res)
    idx += 1
    if idx == 10:
        break;

print "stock num：", idx
print Final.head(10)
print "-" * 20, "Muti-stock table", "-" * 20

pivot_columns = Final.columns[:-1]
# P = Final.pivot_table(index=Final.index,columns='ticker',values=pivot_columns) # Make a pivot table from the data
P = Final.pivot_table(index=Final.index, columns='ticker', values=['c_2_o', 'h_2_o', 'l_2_o', 'c_2_h', 'h_2_l', 'c1_c0',
                                                                   'vol'])  # Make a pivot table from the data
print P.head(10)
print "-" * 20, "Pivot muti-stock table", "-" * 20

mi = P.columns.tolist()
new_ind = pd.Index(e[1] + '_' + e[0] for e in mi)
P.columns = new_ind
P = P.sort_index(axis=1)  # Sort by columns
print P.head(10)
print "-" * 20, "Flat stock index", "-" * 20

clean_and_flat = P.dropna(1)  # 去掉0列？
print clean_and_flat.head(10)
target_cols = list(filter(lambda x: 'c1_c0' in x, clean_and_flat.columns.values))
input_cols = list(filter(lambda x: 'c1_c0' not in x, clean_and_flat.columns.values))
print input_cols
print "Input cols ... "
print target_cols
print "Target cols ..."
InputDF = clean_and_flat[input_cols][:3900]
TargetDF = clean_and_flat[target_cols][:3900]
print InputDF.head(10)
print "InputDF ..."
print TargetDF.head(10)
print "TargetDF ..."
corrs = TargetDF.corr()

num_stocks = len(TargetDF.columns)

print "num stocks :", num_stocks
# print np.exp(TargetDF)
# print (1-np.exp(TargetDF))
# print (1-np.exp(TargetDF)).sum(1)
# TotalReturn = ((1-np.exp(TargetDF)).sum(1))/num_stocks # If i put one dollar in each stock at the close, this is how much I'd get back
# TotalReturn = (1-np.exp(TargetDF))
# print TotalReturn.head(10)
# print "-" * 20,"Return of all stock avg? for current day","-" * 20


'''
#天盈利千分之3，亏损千分之4.6 ？
def labeler(x):
    #if x>0.0029:
    #print x
    x[x > np.log(1.005)] = 1
    x[x < np.log(0.995)] = -1
    x[(x != 1) & (x != -1)] = 0
    #x[x != -1 ] = 0
    #print "--------------------?"
    #print x
    return x

    if x > np.log(1.005):
        return x[x > np.log(1.005)]
    #if x<-0.00462:
    if x < np.log(0.995):
        return x[x < np.log(0.995)]
    else:
        return 0

#MyLabeled = pd.DataFrame()
MyLabeled = TotalReturn.apply(labeler,1)
print MyLabeled

Labeled = pd.DataFrame()
#Labeled['return'] = TotalReturn
#Labeled['class'] = TotalReturn.apply(labeler,1)
#print Labeled['class'].head(10)
print "-" * 20,"Label by return of day","-" * 20




#Labeled['multi_class'] = pd.qcut(TotalReturn,3,labels=range(3))
#Labeled['multi_class'] = pd.qcut(TotalReturn,11,labels=range(11))
#print Labeled['multi_class'].head(10)
#print "-" * 20,"Labeled of multi class","-" * 20

#Labeled['multi_class'] = Labeled['class'] + 1
#Labeled['multi_class'] = MyLabeled + 1

#print Labeled['multi_class'].head(10)
print "-" * 20,"Labeled of multi class","-" * 20
'''
'''
到底应该怎么分类 ？
排序分陈个11个桶，然后5个卖，5个买，一个不动显然不合理！ 必须规定实际return来定买和卖，相对顺序是不合适的？
！！！
使用相对顺序而非绝对值，是为了泛化模型吗？
'''

# print Labeled['class'].value_counts()
print "-" * 20, "Label distribution", "-" * 20
# print Labeled['multi_class'].value_counts()
print "-" * 20, "Qcut multi class distribution", "-" * 20

# Labeled['act_return'] = Labeled['class'] * Labeled['return']                                                            #实际回报是涨跌都取绝对值了，也就是说跌也赚钱的假设吗？
# print Labeled['act_return'].head(10)
print "-" * 20, "Acture return from labeled class", "-" * 20

# Labeled[['return','act_return']].cumsum().plot(subplots=True)
# print Labeled[['return','act_return']].cumsum().head(10)
print "-" * 20, "Return and acture return cumsum", "-" * 20

test_size = 600
from sklearn import linear_model
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.DEBUG)

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'      #disable ts logging
# tf.logging.set_verbosity(tf.logging.ERROR)

# Labeled['tf_class'] = Labeled['multi_class']
# Labeled['tf_class'] = Labeled['class']
num_features = len(InputDF.columns)

print "*" * 50, "Training a rnn network", "*" * 50

num_features = len(InputDF.columns)
# num_classes = pd.Series(MyLabeled.values.ravel()).nunique()            #???

# train = (InputDF[:-test_size].values,Labeled.tf_class[:-test_size].values)
# val = (InputDF[-test_size:].values,Labeled.tf_class[-test_size:].values)

train = (InputDF[:-test_size].values, TargetDF[:-test_size].values)
val = (InputDF[-test_size:].values, TargetDF[-test_size:].values)

print np.shape(TargetDF[-test_size:].values)
print "Data count:",len(train[0])  # 3300 个股票日？ 股票没有那么多,500个
print "Feather count:", num_features
num_stocks = len(TargetDF.columns)
print "Stocks count:", num_stocks

print np.shape(TargetDF[-test_size:].values), np.shape(InputDF[-test_size:].values)
from tensorflow.contrib.layers.python.layers.initializers import xavier_initializer

RNN_HIDDEN_SIZE = 100
# FIRST_LAYER_SIZE=1000
# SECOND_LAYER_SIZE=250
NUM_LAYERS = 2
BATCH_SIZE = 50
NUM_EPOCHS = 200  # 200
lr = 0.0003
NUM_TRAIN_BATCHES = int(len(train[0]) / BATCH_SIZE)  # 每个epoch的批次数量 ， BATCH_SIZE相当于前进步常，其总数为66
NUM_VAL_BATCHES = int(len(val[1]) / BATCH_SIZE)
ATTN_LENGTH = 10
beta = 0


def LstmCell():
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(RNN_HIDDEN_SIZE, state_is_tuple=True)
    return lstm_cell


def lstm_model(X, y):
    cell = tf.nn.rnn_cell.MultiRNNCell([LstmCell() for _ in range(NUM_LAYERS)])
    output, _ = tf.nn.dynamic_rnn(
                                  cell,
                                  inputs=tf.expand_dims(X, -1),
                                  dtype=tf.float64,
                                  time_major=False
                                  )
    #output = tf.reshape(output, [-1, RNN_HIDDEN_SIZE])
    output = tf.transpose(output, [1, 0, 2])
    print "imput0 :",X
    print "imput :",tf.expand_dims(X, -1)
    print "0.1", output, output[-1]
    # last = tf.gather(states_series, int(states_series.get_shape()[0]) - 1)                                          #取最后一个输出
    output = output[-1]
    # 通过无激活函数的全连接层计算线性回归，并将数据压缩成一维数组结构
    predictions = tf.contrib.layers.fully_connected(output, num_stocks, None)

    # 将predictions和labels调整统一的shape
    #labels = tf.reshape(y, [-1])
    #predictions = tf.reshape(predictions, [-1])
    labels = y
    predictions = predictions
    print labels
    print predictions
    loss = tf.losses.mean_squared_error(predictions, labels)
    train_op = tf.contrib.layers.optimize_loss(loss, tf.contrib.framework.get_global_step(),
                                               optimizer="Adagrad",
                                               learning_rate=0.1)
    return predictions, loss, train_op



from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
learn = tf.contrib.learn
PRINT_STEPS = 30

# 生成数据
#test_start = TRAINING_EXAMPLES * SAMPLE_GAP
#test_end = (TRAINING_EXAMPLES + TESTING_EXAMPLES) * SAMPLE_GAP
#train = (InputDF[:-test_size].values, TargetDF[:-test_size].values)
#val = (InputDF[-test_size:].values, TargetDF[-test_size:].values)
train_X, train_y = InputDF[:-test_size].values,TargetDF[:-test_size].values
test_X, test_y = InputDF[-test_size:].values,TargetDF[-test_size:].values

# create a lstm instance and validation monitor
#validation_monitor = learn.monitors.ValidationMonitor(test_X, test_y,
#                                                     every_n_steps=PRINT_STEPS,
#                                                     early_stopping_rounds=1000)
validation_monitor = learn.monitors.ValidationMonitor(test_X, test_y,
                                                     every_n_steps=PRINT_STEPS)

# 进行训练
# 封装之前定义的lstm
regressor = SKCompat(learn.Estimator(model_fn=lstm_model, model_dir="Models/model_0",
                                     config=tf.contrib.learn.RunConfig(
                                    save_checkpoints_steps=20,
                                    save_checkpoints_secs=None,
                                    save_summary_steps=40,
                                    )))




print "--------------------------------------------"
print np.shape(train_X)
print np.shape(train_y),type(train_y)
#print tf.reshape(train_y, [-1])
print "--------------------------------------------"




#regressor.fit(train_X, train_y, batch_size=1, steps=1000,
#              monitors=[validation_monitor])
regressor.fit(train_X, train_y,batch_size=50,steps=1000, monitors=[validation_monitor])
# 计算预测值
print "----------fit over,to predict------------"
predicted = [[pred] for pred in regressor.predict(test_X)]
#print predicted
# 计算MS
print "----------predict over,to rmse------------"
rmse = np.sqrt(((predicted - test_y) ** 2).mean(axis=0))
print type(rmse),np.shape(rmse)
#print("Mean Square Error is:%f" % rmse[0])
print "-----------------------test -------------------------"
#print np.shape(test_X),test_X
print np.shape(test_X[-1:])
print "--------------------- predict -----------------------"
p = regressor.predict(test_X[-1:])
print np.shape(p)
print "*" * 20, "Train over", "*" * 20

