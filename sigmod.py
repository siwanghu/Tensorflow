# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 11:22:35 2018

@author: siwanghu
"""
import numpy as np
import tensorflow as tf

x_train=[]
y_train=[]

x_train=np.linspace(-2,2,100)
y_train=np.sin(x_train)+np.random.normal(0,0.01,100)

x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)

w=tf.Variable(tf.random_normal([1],name="weight"))
b=tf.Variable(tf.random_normal([1],name="bias"))

pre=tf.sigmoid(tf.add(tf.multiply(x,w),b))
cost=tf.reduce_sum(tf.pow(pre-y,2.0))
train=tf.train.AdamOptimizer(0.01).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for _ in range(500):
        for _x,_y in zip(x_train,y_train):
            sess.run(train,feed_dict={x:_x,y:_y})
    print(sess.run(w))
    print(sess.run(b))