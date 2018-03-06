import tensorflow as tf

input1 = tf.constant(1.0)
input2 = tf.constant(2.0)
add=tf.add(input1,input2)
mul=tf.mul(input1,add)

with tf.Session() as sess:
    result=sess.run([mul,add])
    print result
