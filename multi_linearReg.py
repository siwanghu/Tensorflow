# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 19:30:16 2018

@author: siwanghu
"""
import tensorflow as tf

x_train = [[1, 2], [2, 1], [2, 3], [3, 5], [1, 3], [4, 2], [7, 3], [4, 5], [11, 3], [8, 7],[1,1],[0,0],[3,3]]
y_train = [[10], [9], [15], [23], [13], [16], [25], [25], [33], [41],[7],[2],[17]]

x=tf.placeholder(tf.float32,[None,2])
y=tf.placeholder(tf.float32,[None,1])

w=tf.Variable(tf.zeros([2,1]))
b=tf.Variable(tf.zeros([1]))

_y=tf.matmul(x,w)+b
cost=tf.reduce_mean(tf.square(_y-y))

train=tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init=tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train,feed_dict={x:x_train,y:y_train})
    print(sess.run(w))
    print(sess.run(b))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    