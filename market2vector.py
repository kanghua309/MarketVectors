# -*- coding: utf-8 -*-
import pandas as pd
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import plot,savefig
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
for f in os.listdir(datapath):
    filepath = os.path.join(datapath,f)
    if filepath.endswith('.csv'):
        Res = make_inputs(filepath)
        Final = Final.append(Res)
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
InputDF = clean_and_flat[input_cols][:3900]
TargetDF = clean_and_flat[target_cols][:3900]
corrs = TargetDF.corr()

num_stocks = len(TargetDF.columns)

TotalReturn = ((1-np.exp(TargetDF)).sum(1))/num_stocks # If i put one dollar in each stock at the close, this is how much I'd get back
print TotalReturn.head(10)
print "-" * 20,"Return of all stock avg? for current day","-" * 20

#天盈利千分之3，亏损千分之4.6 ？
def labeler(x):
    #if x>0.0029:
    if x > np.log(1.003):
        return 1
    #if x<-0.00462:
    if x < np.log(0.9954):
        return -1
    else:
        return 0

Labeled = pd.DataFrame()
Labeled['return'] = TotalReturn
Labeled['class'] = TotalReturn.apply(labeler,1)
print Labeled['class'].head(10)
print "-" * 20,"Label by return of day","-" * 20

Labeled['multi_class'] = pd.qcut(TotalReturn,11,labels=range(11))
print Labeled['multi_class'].head(10)
print "-" * 20,"Labeled of multi class","-" * 20

def labeler_multi(x):
    if x>0.0029:
        return 1
    if x<-0.00462:
        return -1
    else:
        return 0
print Labeled['class'].value_counts()
print "-" * 20,"Label distribution","-" * 20
print Labeled['multi_class'].value_counts()
print "-" * 20,"Qcut multi class distribution","-" * 20


Labeled['act_return'] = Labeled['class'] * Labeled['return'] #实际回报是涨跌都取绝对值了，也就是说跌也赚钱的假设吗？
print Labeled['act_return'].head(10)
print "-" * 20,"Acture return from labeled class","-" * 20

Labeled[['return','act_return']].cumsum().plot(subplots=True)
print Labeled[['return','act_return']].cumsum().head(10)
print "-" * 20,"Return and acture return cumsum","-" * 20

test_size=600

'''
print "*" * 50,"Data prepare over and todo basic predict","*" * 50

from sklearn import linear_model
logreg = linear_model.LogisticRegression(C=1e5)
res = logreg.fit(InputDF[:-test_size],Labeled['multi_class'][:-test_size]) #取除了最后600个外的数据做测试集,multi class是按收益排序分段泛化的结果集合
print Labeled['multi_class'][:-test_size].head(10)
print InputDF[:-test_size].head(10)
print "-" * 20, "Input & target sample for log regress", "-" * 20

from sklearn.metrics import classification_report,confusion_matrix #验证？
print(classification_report(Labeled['multi_class'][-test_size:],res.predict(InputDF[-test_size:]))) #使用最后600个做测试
print "-" * 20,"lassification report","-" * 20
print(confusion_matrix(Labeled['multi_class'][-test_size:],res.predict(InputDF[-test_size:])))
print res.predict(InputDF)
print "-" * 20,"Confusion matrix 1","-" * 20
Labeled['predicted_action'] = list(map(lambda x: -1 if x <5 else 0 if x==5 else 1,res.predict(InputDF))) #分三段
print Labeled['predicted_action'].head(10)
print "-" * 20,"redicted_action","-" * 20
print(confusion_matrix(Labeled['class'][-test_size:],Labeled['predicted_action'][-test_size:]))
print "-" * 20,"Confusion matrix 2","-" * 20
Labeled['pred_return'] = Labeled['predicted_action'] * Labeled['return']
print Labeled['pred_return'].head(10)
print "-" * 20,"Predict return","-" * 20

Res = Labeled[-test_size:][['return','act_return','pred_return']].cumsum()
print Res.head(10)
print "-" * 20,"Predict return & act_return & pred_return cumsum ","-" * 20

Res[0] =0
Res.plot()
'''

print "*" * 50,"Data prepare over and todo DNN predict","*" * 50

import tensorflow as tf
from  tensorflow.contrib.learn.python.learn.estimators.dnn  import DNNClassifier
from tensorflow.contrib.layers import real_valued_column
Labeled['tf_class'] = Labeled['multi_class']
num_features = len(InputDF.columns)
dropout=0.2
hidden_1_size = 1000
hidden_2_size = 250
num_classes = Labeled.tf_class.nunique()
NUM_EPOCHS=100
BATCH_SIZE=50
lr=0.0001

train = (InputDF[:-test_size].values,Labeled.tf_class[:-test_size].values)
val = (InputDF[-test_size:].values,Labeled.tf_class[-test_size:].values)
NUM_TRAIN_BATCHES = int(len(train[0])/BATCH_SIZE)
NUM_VAL_BATCHES = int(len(val[1])/BATCH_SIZE)

print len(InputDF)


class Model():
    def __init__(self):
        global_step = tf.contrib.framework.get_or_create_global_step()
        self.input_data = tf.placeholder(dtype=tf.float32, shape=[None, num_features])
        self.target_data = tf.placeholder(dtype=tf.int32, shape=[None])
        self.dropout_prob = tf.placeholder(dtype=tf.float32, shape=[])
        with tf.variable_scope("ff"):
            droped_input = tf.nn.dropout(self.input_data, keep_prob=self.dropout_prob)

            layer_1 = tf.contrib.layers.fully_connected(
                num_outputs=hidden_1_size,
                inputs=droped_input,
            )
            layer_2 = tf.contrib.layers.fully_connected(
                num_outputs=hidden_2_size,
                inputs=layer_1,
            )
            self.logits = tf.contrib.layers.fully_connected(
                num_outputs=num_classes,
                activation_fn=None,
                inputs=layer_2,
            )
        with tf.variable_scope("loss"):
            self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.target_data)
            mask = (1 - tf.sign(1 - self.target_data))  # Don't give credit for flat days
            mask = tf.cast(mask, tf.float32)
            self.loss = tf.reduce_sum(self.losses)

        with tf.name_scope("train"):
            opt = tf.train.AdamOptimizer(lr)
            gvs = opt.compute_gradients(self.loss)
            self.train_op = opt.apply_gradients(gvs, global_step=global_step)

        with tf.name_scope("predictions"):
            self.probs = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(self.probs, 1)
            correct_pred = tf.cast(tf.equal(self.predictions, tf.cast(self.target_data, tf.int64)), tf.float64)
            self.accuracy = tf.reduce_mean(correct_pred)


with tf.Graph().as_default():
    model = Model()
    input_ = train[0]
    target = train[1]
    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run([init])
        epoch_loss = 0
        for e in range(NUM_EPOCHS):
            if epoch_loss > 0 and epoch_loss < 1:
                break
            epoch_loss = 0
            for batch in range(0, NUM_TRAIN_BATCHES):
                start = batch * BATCH_SIZE
                end = start + BATCH_SIZE
                feed = {
                    model.input_data: input_[start:end],
                    model.target_data: target[start:end],
                    model.dropout_prob: 0.9
                }

                _, loss, acc = sess.run(
                    [
                        model.train_op,
                        model.loss,
                        model.accuracy,
                    ]
                    , feed_dict=feed
                )
                epoch_loss += loss
            print('step - {0} loss - {1} acc - {2}'.format((1 + batch + NUM_TRAIN_BATCHES * e), epoch_loss, acc))

        print('done training')
        final_preds = np.array([])
        final_probs = None
        for batch in range(0, NUM_VAL_BATCHES):

            start = batch * BATCH_SIZE
            end = start + BATCH_SIZE
            feed = {
                model.input_data: val[0][start:end],
                model.target_data: val[1][start:end],
                model.dropout_prob: 1
            }

            acc, preds, probs = sess.run(
                [
                    model.accuracy,
                    model.predictions,
                    model.probs
                ]
                , feed_dict=feed
            )
            print(acc)
            final_preds = np.concatenate((final_preds, preds), axis=0)
            if final_probs is None:
                final_probs = probs
            else:
                final_probs = np.concatenate((final_probs, probs), axis=0)
        prediction_conf = final_probs[np.argmax(final_probs, 1)]