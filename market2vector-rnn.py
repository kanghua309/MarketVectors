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
    if idx == 10000:
        break;
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

TotalReturn = ((1-np.exp(TargetDF)).sum(1))/num_stocks # If i put one dollar in each stock at the close, this is how much I'd get back
print TotalReturn.head(10)
print "-" * 20,"Return of all stock avg? for current day","-" * 20

#天盈利千分之3，亏损千分之4.6 ？
def labeler(x):
    #if x>0.0029:
    if x > np.log(1.005):
        return 1
    #if x<-0.00462:
    if x < np.log(0.995):
        return -1
    else:
        return 0

Labeled = pd.DataFrame()
Labeled['return'] = TotalReturn
Labeled['class'] = TotalReturn.apply(labeler,1)
print Labeled['class'].head(10)
print "-" * 20,"Label by return of day","-" * 20




#Labeled['multi_class'] = pd.qcut(TotalReturn,3,labels=range(3))
Labeled['multi_class'] = pd.qcut(TotalReturn,11,labels=range(11))
print Labeled['multi_class'].head(10)
print "-" * 20,"Labeled of multi class","-" * 20

#Labeled['multi_class'] = Labeled['class'] + 1
#print Labeled['multi_class'].head(10)
#print "-" * 20,"Labeled of multi class","-" * 20

'''
到底应该怎么分类 ？
排序分陈个11个桶，然后5个卖，5个买，一个不动显然不合理！ 必须规定实际return来定买和卖，相对顺序是不合适的？
！！！
使用相对顺序而非绝对值，是为了泛化模型吗？
'''


print Labeled['class'].value_counts()
print "-" * 20,"Label distribution","-" * 20
print Labeled['multi_class'].value_counts()
print "-" * 20,"Qcut multi class distribution","-" * 20


Labeled['act_return'] = Labeled['class'] * Labeled['return']                                                            #实际回报是涨跌都取绝对值了，也就是说跌也赚钱的假设吗？
print Labeled['act_return'].head(10)
print "-" * 20,"Acture return from labeled class","-" * 20

Labeled[['return','act_return']].cumsum().plot(subplots=True)
print Labeled[['return','act_return']].cumsum().head(10)
print "-" * 20,"Return and acture return cumsum","-" * 20

test_size=600
from sklearn import linear_model
from sklearn.metrics import classification_report,confusion_matrix



import tensorflow as tf


Labeled['tf_class'] = Labeled['multi_class']
#Labeled['tf_class'] = Labeled['class']
num_features = len(InputDF.columns)
dropout=0.5 #0.2
hidden_1_size = 1000
hidden_2_size = 250
num_classes = Labeled.tf_class.nunique()
NUM_EPOCHS=200 #200
BATCH_SIZE=50
lr=0.0001

train = (InputDF[:-test_size].values,Labeled.tf_class[:-test_size].values)
val = (InputDF[-test_size:].values,Labeled.tf_class[-test_size:].values)
NUM_TRAIN_BATCHES = int(len(train[0])/BATCH_SIZE)
NUM_VAL_BATCHES = int(len(val[1])/BATCH_SIZE)

print len(InputDF),num_classes,num_features
print Labeled['tf_class'].head(10)


print "*" * 50,"Training a rnn network","*" * 50

from tensorflow.contrib.layers.python.layers.initializers import xavier_initializer
RNN_HIDDEN_SIZE=100
FIRST_LAYER_SIZE=1000
SECOND_LAYER_SIZE=250
NUM_LAYERS=3
BATCH_SIZE=50
NUM_EPOCHS=200 #200
lr=0.0003
NUM_TRAIN_BATCHES = int(len(train[0])/BATCH_SIZE)
NUM_VAL_BATCHES = int(len(val[1])/BATCH_SIZE)
ATTN_LENGTH=30
beta=0


class RNNModel():
    def __init__(self):
        global_step = tf.contrib.framework.get_or_create_global_step()
        self.input_data = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, num_features],name = "input-data")
        self.target_data = tf.placeholder(dtype=tf.int32, shape=[BATCH_SIZE],name = "target-data")
        self.dropout_prob = tf.placeholder(dtype=tf.float32, shape=[],name = "my-dropout")

        def makeGRUCells():
            cells = []
            for i in range(NUM_LAYERS):
                cell = tf.contrib.rnn.GRUCell(num_units=RNN_HIDDEN_SIZE, )

                if len(cells)== 0:
                    # Add attention wrapper to first layer.
                    cell = tf.contrib.rnn.AttentionCellWrapper(
                        cell, attn_length=ATTN_LENGTH, state_is_tuple=False)
                # attention 很奇怪，加入后state_size: 12600 # 一层4400 - 不加入就300
                #必须false，True 比错

                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.5)
                cells.append(cell)

            attn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=False)  #必须false，True 比错

            #layered_cell = tf.nn.rnn_cell.MultiRNNCell(
            #    [tf.nn.rnn_cell.GRUCell(num_units=RNN_HIDDEN_SIZE, ) for _ in range(NUM_LAYERS)], state_is_tuple=False) #GRU 和 LSTM 效果近似，速度更快点

            #attn_cell = tf.contrib.rnn.AttentionCellWrapper(cell=layered_cell,
            #                                                attn_length=ATTN_LENGTH,                                    # Add attention wrapper to first layer ??
            #                                                state_is_tuple=False)
            return attn_cell  # DropoutWrapper 还没加


        '''
           cells = []
            for i in range(n_layers):                   
                cell = tf.contrib.rnn.LSTMCell(n_hidden, state_is_tuple=True)
                cell = tf.contrib.rnn.AttentionCellWrapper(
                    cell, attn_length=40, state_is_tuple=True)
                cell = tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=0.5)
                cells.append(cell)
            cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            init_state = cell.zero_state(batch_size, tf.float32)
        '''
        self.gru_cell = makeGRUCells()
        #self.zero_state = self.gru_cell.zero_state(1, tf.float32)
        self.zero_state = self.gru_cell.zero_state(BATCH_SIZE, tf.float32)

        self.start_state = tf.placeholder(dtype=tf.float32,
                                          shape=[BATCH_SIZE, self.gru_cell.state_size ],  # (50*100)
                                          name="state")
        #self.start_state = tf.placeholder(dtype=tf.float32,
        #                                  shape=[BATCH_SIZE, RNN_HIDDEN_SIZE * NUM_LAYERS],                             #(50*300)
        #                                  name = "state")
        #self.start_state = tf.placeholder(dtype=tf.float32, shape=[1, self.gru_cell.state_size])

        '''
        with tf.variable_scope("ff", initializer=xavier_initializer(uniform=False)):
            droped_input = tf.nn.dropout(self.input_data, keep_prob=self.dropout_prob)

                                                                                             #no use
            layer_1 = tf.contrib.layers.fully_connected(
                num_outputs=FIRST_LAYER_SIZE,
                inputs=droped_input,

            )
            layer_2 = tf.contrib.layers.fully_connected(
                num_outputs=RNN_HIDDEN_SIZE,
                inputs=layer_1,

            )
        '''
        #split_inputs = tf.reshape(droped_input, shape=[1, BATCH_SIZE, num_features],name="reshape_l1")
        #print split_inputs


        rnn_inputs = tf.reshape(self.input_data, (BATCH_SIZE, num_features))

        #X = tf.reshape(self.input_data, [-1, BATCH_SIZE, self.gru_cell.state_size])
        print self.input_data
        print self.target_data
        print tf.expand_dims(self.input_data, -1)
        print "zero state:",self.zero_state         #Tensor("MultiRNNCellZeroState/MultiRNNCellZeroState/zeros:0", shape=(50, 300), dtype=float32)
        print "start state:",self.start_state
        print "state_size:",self.gru_cell.state_size              # NUM_LAYERS * RNN_HIDDEN_SIZE

        states_series, current_state = tf.nn.dynamic_rnn(self.gru_cell,
                                                         inputs=tf.expand_dims(self.input_data, -1),
                                                         dtype=tf.float32
                                                         )

        '''
        split_inputs = tf.reshape(droped_input, shape=[1, BATCH_SIZE, num_features],
                                  name="reshape_l1")  # Each item in the batch is a time step, iterate through them
        split_inputs = tf.unstack(split_inputs, axis=1, name="unpack_l1")
        print(len(split_inputs),split_inputs)

        states = []
        outputs = []
        with tf.variable_scope("rnn", initializer=xavier_initializer(uniform=False)) as scope:                          #没有使用rnn这种helper函数建立 , 所以还需要通过一个循环逐次对batch中的每一行进行处理
            state = self.start_state
            for i, inp in enumerate(split_inputs):
                if i > 0:
                    scope.reuse_variables()
                output, state = self.gru_cell(inp, state)
                states.append(state)
                outputs.append(output)
        self.end_state = states[-1]
        outputs = tf.stack(outputs, axis=1)  # Pack them back into a single tensor
        outputs = tf.reshape(outputs, shape=[BATCH_SIZE, RNN_HIDDEN_SIZE])
        '''
        print "output0：",states_series         #(50,42,100)                                                            #shape=(50, 42, 100)

        states_series = tf.transpose(states_series, [1, 0, 2])
        print "0.1",states_series
        last = tf.gather(states_series, int(states_series.get_shape()[0]) - 1)                                          #取最后一个输出
        print "last:",last

        #states_series = tf.reshape(states_series, [-1,RNN_HIDDEN_SIZE])

        #states_series = tf.reshape(states_series, shape=[BATCH_SIZE, RNN_HIDDEN_SIZE])
        outputs = last
        self.end_state = current_state
        print "output1:",outputs
        print "state:",current_state,self.end_state

        self.logits = tf.contrib.layers.fully_connected(                                                                #最后还用一个全连接输出？
            num_outputs=num_classes,
            inputs=outputs,
            activation_fn=None
        )

        with tf.variable_scope("loss"):
            self.penalties = tf.reduce_sum([beta * tf.nn.l2_loss(var) for var in tf.trainable_variables()])
            self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.target_data)
            self.loss = tf.reduce_sum(self.losses + beta * self.penalties)

        with tf.name_scope("train_step"):
            opt = tf.train.AdamOptimizer(lr)
            gvs = opt.compute_gradients(self.loss)
            self.train_op = opt.apply_gradients(gvs, global_step=global_step)

        with tf.name_scope("predictions"):
            probs = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(probs, 1)
            correct_pred = tf.cast(tf.equal(self.predictions, tf.cast(self.target_data, tf.int64)), tf.float64)
            self.accuracy = tf.reduce_mean(correct_pred)


with tf.Graph().as_default():

    model = RNNModel()

    input_ = train[0]
    target = train[1]
    with tf.Session(config=tf.ConfigProto(log_device_placement=False,allow_soft_placement=True)) as sess:
        init = tf.global_variables_initializer()
        sess.run([init])
        loss = 2000

        for e in range(NUM_EPOCHS):
            state = sess.run(model.zero_state)
            #print state
            #print "--------------------------------------"
            epoch_loss = 0
            for batch in range(0, NUM_TRAIN_BATCHES):
                start = batch * BATCH_SIZE
                end = start + BATCH_SIZE
                print "-------------",start,end
                feed = {
                    model.input_data: input_[start:end],
                    model.target_data: target[start:end],
                    model.dropout_prob: 0.5,
                    model.start_state: state
                }
                _, loss, acc, state = sess.run(
                    [
                        model.train_op,
                        model.loss,
                        model.accuracy,
                        model.end_state
                    ]
                    , feed_dict=feed
                )
                #print "run state ：",type(state),len(state),state

                epoch_loss += loss

            print('step - {0} loss - {1} acc - {2}'.format((e), epoch_loss, acc))
        final_preds = np.array([])
        for batch in range(0, NUM_VAL_BATCHES):
            start = batch * BATCH_SIZE
            end = start + BATCH_SIZE
            feed = {
                model.input_data: val[0][start:end],
                model.target_data: val[1][start:end],
                model.dropout_prob: 1,
                model.start_state: state
            }
            acc, preds, state = sess.run(
                [
                    model.accuracy,
                    model.predictions,
                    model.end_state
                ]
                , feed_dict=feed
            )
            print(acc)
            assert len(preds) == BATCH_SIZE
            final_preds = np.concatenate((final_preds, preds), axis=0)

print "*" * 20,"Train over","*" * 20

'''
Result['rnn_pred'] = final_preds
#Result['mod_rnn_prod'] = list(map(lambda x: -1 if x <1 else 0 if x==1 else 1,final_preds))
Result['mod_rnn_prod'] = list(map(lambda x: -1 if x <5 else 0 if x==5 else 1,final_preds))


Result['rnn_ret'] = Result.mod_rnn_prod*Result['return']

print(confusion_matrix(Result['multi_class'],Result['rnn_pred']))
print "-" * 20,"Confusion matrix","-" * 20

print(classification_report(Result['class'],Result['mod_rnn_prod']))
print "-" * 20,"Classification report","-" * 20

print(confusion_matrix(Result['class'],Result['mod_rnn_prod']))
print "-" * 20,"Confusion matrix2","-" * 20

Res = (Result[-test_size:][['return','nn_ret','rnn_ret','pred_return']]).cumsum()
print Res.tail(10)
print "-" * 20,"Return & nn_ret & rnn_ret & pred_return cumsum","-" * 20

Res[0] =0
Res.plot(figsize=(20,10))

Res.columns =['Market Baseline','Simple Neural Newtwork','My Algo','Logistic Regression (simple ML)','Do Nothing(0)']
Res.plot(figsize=(20,10),title="Performance of MarketVectors algo over 27 months compared with baselines")


Res.columns
Res.columns =['baseline','logistic_regression','feed_forward_net','rnn_net','do_nothing']
Res.plot(figsize=(20,10))
'''