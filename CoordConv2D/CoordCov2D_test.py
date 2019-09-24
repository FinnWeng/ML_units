import tensorflow as tf
import numpy as np


class CoordConv2D:
    def __init__(self, with_r = False):
        self.with_r = with_r
    def __call__(self,input):
        self.x_dim = input.shape.as_list()[2]
        self.y_dim = input.shape.as_list()[1]
        batch_size_tensor = tf.shape(input)[0]
        xy_vector = tf.ones([self.y_dim,1])
        yx_vector = tf.ones([1,self.x_dim])
        x_range = tf.reshape(tf.range(1,self.x_dim+1,1,dtype=tf.float32),[1,self.x_dim])
        y_range = tf.reshape(tf.range(1,self.y_dim+1,1,dtype=tf.float32),[self.y_dim,1])
        x_normal_range = tf.multiply(x_range,1/self.x_dim)
        y_normal_range = tf.multiply(y_range,1/self.y_dim)
        x_mat = tf.matmul(xy_vector,x_normal_range)
        y_mat = tf.matmul(y_normal_range,yx_vector)

        x_mat = tf.reshape(x_mat,[1,self.y_dim,self.x_dim,1])
        y_mat = tf.reshape(y_mat,[1,self.y_dim,self.x_dim,1])
        x_mats = tf.tile(x_mat,[batch_size_tensor,1,1,1])
        y_mats = tf.tile(y_mat,[batch_size_tensor,1,1,1])


        
        if self.with_r == True:
            # # orgin
            # r = ((x_mats-0.5)**2 + (y_mats-0.5)**2)
            # r = tf.sqrt(r)

            # I test 
            r = (tf.sqrt((x_mats-0.5)**2) + tf.sqrt((y_mats-0.5)**2))

            input = tf.concat([input,x_mats,y_mats,r],axis=-1)
            return input
        else:
            input = tf.concat([input,x_mats,y_mats],axis=-1)
            return input


a = tf.placeholder(dtype=tf.float32,shape=[10,10,20,8])
b = tf.keras.layers.Conv2D(5,3,1,"same")(a)
b = tf.keras.layers.Conv2D(5,3,1,"same")(b)
b = tf.keras.layers.Conv2D(5,3,1,"same",activation="sigmoid")(b)
b = tf.keras.layers.Conv2D(5,3,1,"same",activation="sigmoid")(b) 
b = CoordConv2D(with_r = False)(b)








with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(b,feed_dict={a:np.arange(16000).reshape([10,10,20,8])*100})
    print(result[:,:,:,-1])

    