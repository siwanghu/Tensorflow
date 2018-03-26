# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 11:58:32 2018

@author: siwanghu
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

inputs=28
times=28

lstm_size=100
classes=10
batch_size=100
batch=mnist.train.num_examples//batch_size

x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])

weights=tf.Variable(tf.truncated_normal([lstm_size,classes],stddev=0.1))
biases=tf.Variable(tf.constant(0.1,shape=[classes]))

def RNN(x,weights,biases):
    x_inputs=tf.reshape(x,[-1,times,inputs])
    lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    outputs,state=tf.nn.dynamic_rnn(lstm_cell,x_inputs,dtype=tf.float32)
    return tf.nn.softmax(tf.matmul(state[1],weights)+biases)

pre=RNN(x,weights,biases)

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pre,labels=y))
train=tf.train.AdamOptimizer(1e-4).minimize(loss)

correct=tf.equal(tf.argmax(y,1),tf.argmax(pre,1))
accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))

init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(10):
        for _ in range(batch):
            xs,ys=mnist.train.next_batch(batch_size)
            sess.run(train,feed_dict={x:xs,y:ys})
        acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("step: "+str(i)+" acc:"+str(acc))