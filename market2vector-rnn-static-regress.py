# -*- coding: utf-8 -*-
import pandas as pd
import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import plot,savefig
from matplotlib import pyplot as plt


pd.set_option('display.width', 800)


datapath = './daily'
filepath = os.path.join(datapath,os.listdir('./daily')[0])

import re
ticker_regex = re.compile('.+_(?P<ticker>.+)\.csv')
get_ticker =lambda x :ticker_regex.match(x).groupdict()['ticker']
print(filepath,get_ticker(filepath))

ret = lambda x,y: np.log(y/x) #Log return
zscore = lambda x:(x -x.mean())/x.std() # zscore


D = pd.read_csv(filepath,header=None,names=['UNK','o','h','l','c','v']) #Load the dataframe with headers
print D.head(10)
print "-" * 20,"Example csv data","-" * 20

# 各指标都是和open价格做比值，在标准化分
def make_inputs(filepath):
    D = pd.read_csv(filepath,header=None,names=['UNK','o','h','l','c','v']) #Load the dataframe with headers
    D.index = pd.to_datetime(D.index,format='%Y%m%d') # Set the indix to a datetime
    Res = pd.DataFrame()
    ticker = get_ticker(filepath)

    Res['c_2_o'] = zscore(ret(D.o,D.c))
    Res['h_2_o'] = zscore(ret(D.o,D.h))
    Res['l_2_o'] = zscore(ret(D.o,D.l))
    Res['c_2_h'] = zscore(ret(D.h,D.c))
    Res['h_2_l'] = zscore(ret(D.h,D.l))
    Res['c1_c0'] = ret(D.c,D.c.shift(-1)).fillna(0) #Tommorows return
    Res['vol'] = zscore(D.v)
    Res['ticker'] = ticker
    return Res
Res = make_inputs(filepath)
#print Res.head()
#print Res.corr()

Final = pd.DataFrame()
idx=0
for f in os.listdir(datapath):
    filepath = os.path.join(datapath,f)
    if filepath.endswith('.csv'):
        Res = make_inputs(filepath)
        Final = Final.append(Res)
    idx += 1
    if idx == 10:
        break;
print "stock num：",idx
print Final.head(10)
print "-" * 20,"Muti-stock table","-" * 20


pivot_columns = Final.columns[:-1]
#P = Final.pivot_table(index=Final.index,columns='ticker',values=pivot_columns) # Make a pivot table from the data
P = Final.pivot_table(index=Final.index,columns='ticker',values=['c_2_o','h_2_o','l_2_o','c_2_h','h_2_l','c1_c0','vol']) # Make a pivot table from the data
print P.head(10)
print "-" * 20,"Pivot muti-stock table","-" * 20

mi = P.columns.tolist()
new_ind = pd.Index(e[1] +'_' + e[0] for e in mi)
P.columns = new_ind
P = P.sort_index(axis=1) # Sort by columns
print P.head(10)
print "-" * 20,"Flat stock index","-" * 20

clean_and_flat = P.dropna(1) #去掉0列？
print clean_and_flat.head(10)
target_cols = list(filter(lambda x: 'c1_c0' in x, clean_and_flat.columns.values))
input_cols  = list(filter(lambda x: 'c1_c0' not in x, clean_and_flat.columns.values))
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
#print np.exp(TargetDF)
#print (1-np.exp(TargetDF))
#print (1-np.exp(TargetDF)).sum(1)
#TotalReturn = ((1-np.exp(TargetDF)).sum(1))/num_stocks # If i put one dollar in each stock at the close, this is how much I'd get back
#TotalReturn = (1-np.exp(TargetDF))
#print TotalReturn.head(10)
#print "-" * 20,"Return of all stock avg? for current day","-" * 20


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


#print Labeled['class'].value_counts()
print "-" * 20,"Label distribution","-" * 20
#print Labeled['multi_class'].value_counts()
print "-" * 20,"Qcut multi class distribution","-" * 20


#Labeled['act_return'] = Labeled['class'] * Labeled['return']                                                            #实际回报是涨跌都取绝对值了，也就是说跌也赚钱的假设吗？
#print Labeled['act_return'].head(10)
print "-" * 20,"Acture return from labeled class","-" * 20

#Labeled[['return','act_return']].cumsum().plot(subplots=True)
#print Labeled[['return','act_return']].cumsum().head(10)
print "-" * 20,"Return and acture return cumsum","-" * 20

test_size=600
from sklearn import linear_model
from sklearn.metrics import classification_report,confusion_matrix


import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'      #disable ts logging
#tf.logging.set_verbosity(tf.logging.ERROR)

#Labeled['tf_class'] = Labeled['multi_class']
#Labeled['tf_class'] = Labeled['class']
num_features = len(InputDF.columns)



print "*" * 50,"Training a rnn network","*" * 50

num_features = len(InputDF.columns)
#num_classes = pd.Series(MyLabeled.values.ravel()).nunique()            #???

#train = (InputDF[:-test_size].values,Labeled.tf_class[:-test_size].values)
#val = (InputDF[-test_size:].values,Labeled.tf_class[-test_size:].values)

train = (InputDF[:-test_size].values,TargetDF[:-test_size].values)
val = (InputDF[-test_size:].values,TargetDF[-test_size:].values)
print np.shape(TargetDF[-test_size:].values)

print len(train[0])   #3300 个股票日？ 股票没有那么多,500个
print "Feather count:",num_features
num_stocks = len(TargetDF.columns)
print "Stocks count:",num_stocks

print np.shape(TargetDF[-test_size:].values),np.shape(InputDF[-test_size:].values)
from tensorflow.contrib.layers.python.layers.initializers import xavier_initializer
RNN_HIDDEN_SIZE=100
#FIRST_LAYER_SIZE=1000
#SECOND_LAYER_SIZE=250
NUM_LAYERS=2
BATCH_SIZE=50
NUM_EPOCHS=2 #200
lr=0.0003
NUM_TRAIN_BATCHES = int(len(train[0])/BATCH_SIZE)         #每个epoch的批次数量 ， BATCH_SIZE相当于前进步常，其总数为66
NUM_VAL_BATCHES = int(len(val[1])/BATCH_SIZE)
ATTN_LENGTH=30
beta=0

#print NUM_TRAIN_BATCHES  #66
class RNNModel():
    def __init__(self):
        global_step = tf.contrib.framework.get_or_create_global_step()
        self.input_data = tf.placeholder(dtype=tf.float32, shape=[None, num_features],name = "input-data")
        self.target_data = tf.placeholder(dtype=tf.float32, shape=[None,  num_stocks],name = "target-data")           #------------------------------????????????????????????????
        self.dropout_prob = tf.placeholder(dtype=tf.float32, shape=[],name = "my-dropout")

        def makeGRUCells():
            cells = []
            for i in range(NUM_LAYERS):
                cell = tf.nn.rnn_cell.GRUCell(num_units=RNN_HIDDEN_SIZE)  #tf.contrib.rnn 下的rnn 要比tf.nn.rnn_cell下的慢好多，新的优化了？ 初始化加不加有何区别？ 会让收敛快点吗？
                if len(cells)== 0:
                    # Add attention wrapper to first layer.
                    cell = tf.contrib.rnn.AttentionCellWrapper(
                       cell, attn_length=ATTN_LENGTH, state_is_tuple=False)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell,output_keep_prob=0.5)
                cells.append(cell)
            attn_cell =  tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=False)  #GRUCell必须false，True 比错 ,如果是BasicLSTMCell 必须True

            return attn_cell

        self.gru_cell = makeGRUCells()
        #self.zero_state = self.gru_cell.zero_state(1, tf.float32)
        #self.zero_state = self.gru_cell.zero_state(BATCH_SIZE, tf.float32)

        print self.input_data
        print self.target_data

        states_series, current_state = tf.nn.dynamic_rnn(self.gru_cell,                                                 #收敛的好慢 相比自己构造层次
                                                         inputs=tf.expand_dims(self.input_data, -1),                    #shape=(42, 50, 100)  ? ("ExpandDims:0", shape=(50, 42, 1), dtype=float32)
                                                         dtype=tf.float32,                                              #由于预测，最后给定的batch 是1，而对于非订场batch ，就要用dynamic rnn？ 而且iniita_state 不能设置，因为shape不定？
                                                         #initial_state=self.zero_state,
                                                         time_major=False                                               #如果 inputs 为 (batches, steps, inputs) ==> time_major=False
                                                         )
        '''
        split_inputs = tf.reshape(self.input_data, shape=[1, BATCH_SIZE, num_features],
                                  name="reshape_l1")  # Each item in the batch is a time step, iterate through them
        print split_inputs
        x = tf.unstack(split_inputs, axis=1, name="unpack_l1")

        states_series, current_state = tf.nn.static_rnn(self.gru_cell,  # 收敛的好慢 相比自己构造层次
                                                         inputs=x,  #这个输入是一个list，len（list）= 步长 list[0].shape=[batch,input] ?
                                                         dtype=tf.float32
                                                         )
        '''
        print "current state：",current_state
        print "output0：",states_series         #(50,42,100)                                                            #shape=(50, 42, 100)
        states_series = tf.transpose(states_series, [1, 0, 2])
        print "0.1",states_series
        last = tf.gather(states_series, int(states_series.get_shape()[0]) - 1)                                          #取最后一个输出
        print "last:",last
        outputs = last
        print "output1:",outputs
        print "state:",current_state

        self.Y_pred = tf.contrib.layers.fully_connected(
            outputs, num_stocks, activation_fn=None)  # We use the last cell's output

        # cost/loss
        self.loss = tf.reduce_sum(tf.square(self.Y_pred - self.target_data), name='losses_sum')

        # output is result of linear activation of last layer of RNN
        #weight = tf.Variable(tf.random_normal([RNN_HIDDEN_SIZE, num_stocks]))
        #bias = tf.Variable(tf.random_normal([num_stocks]))
        #predictions = tf.matmul(outputs, weight) + bias



        # 2. Define the loss function for training/evaluation
        #print 'targets={}'.format(self.target_data)
        #print 'preds={}'.format(predictions)
        #self.loss = tf.losses.mean_squared_error(self.target_data, predictions)
        #    "rmse": tf.metrics.root_mean_squared_error(self.target_data, predictions)
        #}

        #self.predictions = tf.placeholder(tf.float32, [None, num_stocks], name='predictions')
        #self.rmse = tf.metrics.root_mean_squared_error(self.target_data, self.predictions)
        # 3. Define the training operation/optimizer
        optimizer = tf.train.AdamOptimizer(0.0001)
        self.train = optimizer.minimize(self.loss, name='train')

        # RMSE
        #self.targets = tf.placeholder(tf.float32, [None, 1], name='targets')
        self.predictions = tf.placeholder(tf.float32, [None, num_stocks], name='predictions')
        print self.predictions
        self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.target_data - self.predictions)), name='rmse')

with tf.Graph().as_default():
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.log_device_placement = False
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5

    model = RNNModel()
    input_ = train[0]
    target = train[1]
    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run([init])
        loss = 2000

        for e in range(NUM_EPOCHS):
            state = sess.run(model.zero_state)
            #print state
            #print "--------------------------------------"
            epoch_loss = 0
            for batch in range(0, NUM_TRAIN_BATCHES):      #多个指标（多个股票），按时间其头并进
                start = batch * BATCH_SIZE
                end = start + BATCH_SIZE
                #print "-------------",start,end
                feed = {
                    model.input_data: input_[start:end],
                    model.target_data: target[start:end],
                    model.dropout_prob: 0.5,
                }
                _, loss  = sess.run(
                    [
                        model.train,
                        model.loss,
                    ]
                    , feed_dict=feed
                )

                epoch_loss += loss
            print('step - {0} loss - {1} '.format((e), epoch_loss))

        for batch in range(0, NUM_VAL_BATCHES):
            start = batch * BATCH_SIZE
            end = start + BATCH_SIZE
            feed = {
                model.input_data: val[0][start:end],
                model.target_data: val[1][start:end],
                model.dropout_prob: 1,
                # model.start_state: state
            }

            test_predict = sess.run(
                [
                    model.Y_pred,
                ]
                , feed_dict=feed
            )
            print("predictor:",np.shape(test_predict)),np.shape(test_predict[-1])
            rmse = sess.run(
                model.rmse,
                feed_dict={
                    model.target_data: val[1][start:end],
                    model.predictions: test_predict[-1]
                })

            print("RMSE: {}".format(rmse))


        # Predictions test
        print val[0][-1]
        print np.shape(val[0]),np.shape(val[0][-1:]),np.shape(val[0][-1:].transpose())
        prediction_test = sess.run(
            model.Y_pred,
            feed_dict={
                model.input_data: val[0][-1:]
                })
        print "prediction_test:",prediction_test
        '''
        print (type(val[0][-1]),np.shape(val[0][-1:]))
        print val[0][-1:]
        feed = {
            model.input_data: val[0][-1:],
            #model.target_data: val[1][-1],
            model.dropout_prob: 1,
            # model.start_state: state
        }
        prec = sess.run(
            [
                model.Y_pred,
                # model.rmse,
            ]
            , feed_dict=feed
        )
        print "------------------------------------------------dest day:"
        print(prec)
        '''
            #assert len(preds) == BATCH_SIZE
            #final_preds = np.concatenate((final_preds, preds), axis=0)

print "*" * 20,"Train over","*" * 20

