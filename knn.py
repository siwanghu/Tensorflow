# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 17:36:17 2018

@author: siwanghu
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("./MNIST_data",one_hot=True)

train_x,train_y=mnist.train.next_batch(5000)
test_x,test_y=mnist.test.next_batch(500)

x_train=tf.placeholder("float",[None,784])
x_test=tf.placeholder("float",[784])

dis=tf.reduce_sum(tf.abs(tf.add(x_train,tf.negative(x_test))),reduction_indices=1)
pre=tf.arg_min(dis, 0)

acc=0.0
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(len(test_x)):
        index=sess.run(pre,feed_dict={x_train:train_x,x_test:test_x[i,:]})
        if np.argmax(train_y[index]) == np.argmax(test_y[i]):
            acc+= 1
    print("Done!")
    print("Accuracy:", acc/len(test_x))
    
    
    
    
    
    
    
    
    