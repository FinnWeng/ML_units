import tensorflow as tf
import numpy as np
from tensorflow.keras import layers




class embd_projection(layers.Layer):
    def __init__(self,filters):
        super(embd_projection, self).__init__()
        self.x_d1 = tf.keras.layers.Dense(filters,use_bias=False,name="1st_projection_layer_should_not_repeat")
        self.x_d2 = tf.keras.layers.Dense(filters,use_bias=False,name="2nd_projection_layer_should_not_repeat")
    def call(self,x):
        x = self.x_d1(x)
        x = tf.nn.relu(x)
        x = self.x_d2(x)
        return x

def c2D(x,filters,kernel):
    x = tf.keras.layers.Conv2D(filters, kernel_size=kernel,padding='same')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.leaky_relu(x,alpha=0.2)
    return x

def block(x,filters):
    x = c2D(x,filters,3)
    x = c2D(x,filters,3)
    return x
    
def d1D(x,filters):
    x = tf.keras.layers.Dense(filters)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
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

class model_body_layer(layers.Layer):
    def __init__(self,h_w,class_num,filters,projection):
        super(model_body_layer, self).__init__()
        self.h_w = h_w

        self.class_num = class_num
        self.filters = filters

        self.projection = projection
    def build_body_layer(self):
        self.conv1 = tf.keras.layers.Conv1D(64, kernel_size=3,padding='same',use_bias=False,name = "This_should_not_repeat")
        self.gmp1 = tf.keras.layers.GlobalMaxPool1D()
        self.d1 = tf.keras.layers.Dense(self.class_num,name = "predict")

        
    def call(self,body_input):
        x = self.conv1(body_input)
        x = self.gmp1(x)
        x_embd = self.projection(x)
        logits = self.d1(x)
        return logits, x_embd




class Contrastive_Net:
    def __init__(self,h_w,class_num,filters):
        
        self.h_w = h_w

        self.class_num = class_num
        self.filters = filters

        self.projection_1 = embd_projection(self.filters*8)
    
    def build_model_body(self):
        # with tf.name_scope("Body") as scope:

        body_input = tf.keras.Input(shape=self.h_w,name="body_input",dtype=tf.float32)
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
    
    def build_single_body(self):
        body_input = tf.keras.Input(shape=self.h_w,name="body_input",dtype=tf.float32)
        x = tf.keras.layers.Conv1D(64, kernel_size=3,padding='same',use_bias=False,name = "This_should_not_repeat")(body_input)
        x = tf.keras.layers.GlobalMaxPool1D()(x)
        x_embd = self.projection_1(x)
        logits = tf.keras.layers.Dense(self.class_num,name = "predict")(x)
        model = tf.keras.Model(inputs=body_input,outputs=[logits,x_embd])
        return model
    



    # def custom_loss(self,yt,yp):
    #     x_embd_a,x_embd_b = yp[0],yp[1]


    #     # x_embd_contrasive_loss = compute_contrasive_loss(self.x_embd_a,self.x_embd_b)
    #     x_embd_contrasive_loss = compute_contrasive_loss(x_embd_a,x_embd_b)

    #     return  x_embd_contrasive_loss
    
    # def custom_cata_loss(self,yt,yp):

    #     logits_a,logits_b = yp[0], yp[1]
    #     y_a,y_b = yt[0],yt[1]
    #     logits_concat = tf.concat([logits_a,logits_b],axis=0)
    #     y_concat = tf.concat([y_a,y_b],axis=0)
    #     cata_loss = tf.keras.losses.categorical_crossentropy(logits_concat, y_concat, from_logits = True)
    #     return cata_loss
    
    
        
    
    

    def build(self):   
        
        self.x_a = tf.keras.Input(shape=self.h_w,name="input_a",dtype=tf.float32)
        self.x_b = tf.keras.Input(shape=self.h_w,name="input_b",dtype=tf.float32)
        

        '''
        $ Try reusing by model
        '''

        model_body = self.build_model_body()
        # model_body = self.build_single_body()
        logits_a, x_embd_a = model_body(self.x_a)
        logits_b, x_embd_b = model_body(self.x_b)

        # print("logits_a",logits_a)
        # print("logits_b",logits_b)
        # print("x_embd_a",x_embd_a)
        # print("x_embd_b",x_embd_b)

        '''
        # Try reusing by custom layers. It works, verified by 

        for var in net.variables:
            print(var.name)

        And it shows:
        
        model_body_layer/embd_projection/dense/kernel:0
        model_body_layer/embd_projection/dense_1/kernel:0
        model_body_layer/This_should_not_repeat/kernel:0
        model_body_layer/predict/kernel:0
        model_body_layer/predict/bias:0

        '''
        # body_layer = model_body_layer(self.h_w,self.class_num,self.filters,self.projection_1)
        # body_layer.build_body_layer()

        # logits_a, self.x_embd_a = body_layer(inputs = self.x_a)
        # logits_b, self.x_embd_b = body_layer(inputs = self.x_b)

        
        input_list = [self.x_a,self.x_b]
        # model = tf.keras.Model(inputs = input_list,outputs = [(logits_a,logits_b),(x_embd_a,x_embd_b)], name = "Contrastive_Net")
        concat_embd = tf.concat([x_embd_a,x_embd_b],axis=-1)
        model = tf.keras.Model(inputs = input_list,outputs = [logits_a,logits_b,concat_embd], name = "Contrastive_Net")



        return model
    

