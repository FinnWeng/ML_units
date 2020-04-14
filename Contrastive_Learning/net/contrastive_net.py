import tensorflow as tf
import numpy as np
from tensorflow.keras import layers




class embd_projection(layers.Layer):
    def __init__(self,filters):
        super(embd_projection, self).__init__()
        self.x_d1 = tf.keras.layers.Dense(filters,use_bias=False)
        self.x_d2 = tf.keras.layers.Dense(filters,use_bias=False)
    def call(self,x):
        x = self.x_d1(x)
        x = tf.nn.relu(x)
        x = self.x_d2(x)
        return x

def c2D(x,filters,kernel):
    x = tf.keras.layers.Conv2D(filters, kernel_size=kernel,padding='same',use_bias=False)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.leaky_relu(x,alpha=0.2)
    return x

def block(x,filters):
    x = c2D(x,filters,3)
    x = c2D(x,filters,3)
    return x
    
def d1D(x,filters):
    x = tf.keras.layers.Dense(filters,use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.leaky_relu(x,alpha=0.2)
    return x



def compute_contrasive_loss(vec_a, vec_b):
    '''
    vec: (...,c)
    '''
    vec_a = tf.reshape(vec_a,[-1,vec_a.shape[-1]]) # (n,c)
    vec_b = tf.reshape(vec_b,[-1,vec_b.shape[-1]])
    print("vec_a:",vec_a)
    print("vec_b:",vec_b)
    print("type(vec_a.shape[-1]):",type(vec_a.shape[-1]))
    vec_ab = tf.concat([vec_a,vec_b],axis=0) # (2n,c)
    print("vec_ab:",vec_ab)
    pairwise_cos_sim = tf.math.exp(tf.keras.losses.cosine_similarity(vec_ab,vec_ab)) # # (2n,1)
    print("pairwise_cos_sim:",pairwise_cos_sim)
    sum_vec = tf.reduce_sum(pairwise_cos_sim,axis=0) # (1,)
    print("sum_vec:",sum_vec)

    no_self_matrix = sum_vec - pairwise_cos_sim # (2n,1)
    print("no_self_matrix:",no_self_matrix)
    sim_loss = tf.reduce_sum(tf.math.divide(pairwise_cos_sim,no_self_matrix))
    return sim_loss



class Contrastive_Net:
    def __init__(self,h_w,class_num,filters):
        
        self.h_w = h_w

        self.class_num = class_num
        self.filters = filters

        self.projection_1 = embd_projection(self.filters*8)
    
    def build_model_body(self):

        body_input = tf.keras.Input(shape=self.h_w,name="body_input",dtype=tf.int32)
        expand_body_input = tf.expand_dims(body_input,axis=-1)
        expand_body_input = tf.cast(expand_body_input,tf.float32)/255
        
        
        x = c2D(expand_body_input,self.filters,3)
        x = tf.keras.layers.SpatialDropout2D(0.1)(x)
        x = c2D(x,self.filters,3)
        x = tf.keras.layers.SpatialDropout2D(0.1)(x)
        x = c2D(x,self.filters*2,3)
        x = tf.keras.layers.MaxPool2D(2)(x)
        x = tf.keras.layers.SpatialDropout2D(0.1)(x)

        x = block(x,self.filters*4)
        x = tf.keras.layers.MaxPool2D(2)(x)
        
        
        x = block(x,self.filters*4)
        x = tf.keras.layers.GlobalMaxPool2D()(x)
        print("x = tf.keras.layers.GlobalMaxPool2D()(x):",x)

        x_embd = self.projection_1(x)


        x = d1D(x,128)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = d1D(x,128)
        x = tf.keras.layers.Dropout(0.1)(x)

        logits = tf.keras.layers.Dense(self.class_num,name = "predict")(x)

        # model = tf.keras.Model(inputs=[body_P_input,body_M_input],outputs=[diff_slow,diff_fast,logits,x_space_slow_recon_logit,\
        #     x_d_slow_recon_logit,x_space_fast_recon_logit,x_d_fast_recon_logit,diff_slow,diff_fast, x_M_recon,x_embd])

        model = tf.keras.Model(inputs=body_input,outputs=[logits,x_embd])

        return model
    



    def custom_loss(self):


        x_embd_contrasive_loss = compute_contrasive_loss(self.x_embd_a,self.x_embd_b)

        return  x_embd_contrasive_loss
    
    
        
    
    

    def build(self):   
        
        self.x_a = tf.keras.Input(shape=self.h_w,name="input_a",dtype=tf.int32)
        self.x_b = tf.keras.Input(shape=self.h_w,name="input_b",dtype=tf.int32)
        

        model_body = self.build_model_body()

        logits_a, self.x_embd_a = model_body(inputs = self.x_a)
        logits_b, self.x_embd_b = model_body(inputs = self.x_b)
        
        input_list = [self.x_a,self.x_b]
        model = tf.keras.Model(inputs = input_list,outputs = [logits_a,logits_b], name = "Contrastive_Net")

        return model
    

