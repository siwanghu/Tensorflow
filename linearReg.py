# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 18:37:54 2018

@author: siwanghu
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def createData():
    train_x=np.linspace(-1,1,100)
    train_y=3*train_x+0.2+np.random.randn(*train_x.shape)*0.4
    return train_x,train_y

train_x,train_y=createData()
x=tf.placeholder("float",name="x")
y=tf.placeholder("float",name="y")


with tf.name_scope("model"):
    w=tf.Variable(-0.5,name="w")
    b=tf.Variable(-1.0,name="b")
    y_model=tf.multiply(x,w)+b
    
with tf.name_scope("cost"):
    loss=tf.pow(y-y_model,2)

train=tf.train.GradientDescentOptimizer(0.01).minimize(loss)


sess=tf.Session()
init=tf.initialize_all_variables()
sess.run(init)
for i in range(50):
    for _x,_y in zip(train_x,train_y):
        sess.run(train,feed_dict={x:_x,y:_y})
    _w=w.eval(session=sess)
    _b=b.eval(session=sess)
    plt.figure()
    plt.scatter(train_x,train_y)
    plt.plot(train_x,_b+_w*train_x)

plt.figure()
plt.scatter(train_x,train_y)
plt.plot(train_x,sess.run(b)+train_x*sess.run(w))
print(sess.run(w))
print(sess.run(b)) 