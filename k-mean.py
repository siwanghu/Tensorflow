# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 09:29:41 2018

@author: siwanghu
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def createData():
    dataset=[]
    for i in range(2000):
        if np.random.random()>0.5:
            dataset.append([np.random.normal(0.0,0.9),np.random.normal(0.0,0.9)])
        else:
            dataset.append([np.random.normal(3.0,0.5),np.random.normal(1.0,0.5)])
    return dataset

def show(dataset):
    data=np.asarray(dataset).transpose()
    fig,ax=plt.subplots()
    ax.scatter(data[0],data[1],marker='o',s=1)
    plt.plot()

k=2
data=createData()
vectors=tf.constant(data)
centroid=tf.Variable(tf.slice(tf.random_shuffle(vectors),[0,0],[k,-1]))

exp_vectors=tf.expand_dims(vectors,0)
exp_centroid=tf.expand_dims(centroid,1)

diff=tf.subtract(exp_vectors,exp_centroid)
sqr=tf.square(diff)
dis=tf.reduce_sum(sqr,2)
assign=tf.argmin(dis,0)

means=tf.concat([tf.reduce_mean(tf.gather(vectors,tf.reshape(tf.where(tf.equal(assign,c)),[1,-1])),reduction_indices=[1])for c in range(k)],0)
up_centroid=tf.assign(centroid,means)

sess=tf.Session()
sess.run(tf.initialize_all_variables())

for step in range(100):
    _,centroid_values,assignment_values=sess.run([up_centroid,centroid,assign])

show(data)
print(centroid_values)
























