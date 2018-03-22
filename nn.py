# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 16:49:34 2018

@author: siwanghu
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs,ins,outs,activation_function=None):
    weights=tf.Variable(tf.random_normal([ins,outs]))
    biases=tf.Variable(tf.zeros([1,outs])+0.1)
    plus=tf.matmul(inputs,weights)+biases
    if activation_function is None:
        outputs=plus
    else:
        outputs=activation_function(plus)
    return outputs

x_train=np.linspace(-1,1,200)[:,np.newaxis]
y_train=np.square(x_train)-0.5+np.random.normal(0,0.05,x_train.shape)

x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])

layer1=add_layer(x,1,5,activation_function=tf.nn.relu)
pre=add_layer(layer1,5,1,activation_function=None)

loss=tf.reduce_mean(tf.reduce_sum(tf.square(y-pre),reduction_indices=[1]))
train=tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_train, y_train)
plt.ion()

for i in range(2000):
    sess.run(train,feed_dict={x:x_train,y:y_train})
    if i % 50 == 0:
        print(sess.run(loss, feed_dict={x: x_train, y: y_train}))
        try:
            ax.lines.remove(lines[0])
        except:
            pass
        pre_value = sess.run(pre, feed_dict={x: x_train})
        lines=ax.plot(x_train.reshape(200), pre_value.reshape(200), 'r-', lw=5)
        plt.pause(0.2)


















      