import tensorflow as tf
import tensorflow_addons as tfa
from net.contrastive_net import Contrastive_Net
from tensorflow import keras
from utils.imagenet_loader import MiniImagenetDataset
import matplotlib.pyplot as plt

import os

import numpy as np
import argparse


class Augmentation:
    def __init__(self,IMG_SHAPE):
        self.IMG_SHAPE = IMG_SHAPE

    def tf_noise_aug(self,input_image):
        SNR = 0.8
        max_signal = tf.reduce_max(input_image) # as 1
        input_shape = self.IMG_SHAPE

        mu, sigma = 0.0, 0.1
        noise_vector = tf.random.normal([input_shape[0]*input_shape[1]*input_shape[2]],mu,sigma)
        noise_vector = noise_vector * (max_signal)*(SNR**2)
        noise_mat = tf.reshape(noise_vector,input_shape)

        noisy_img = input_image + noise_mat
        noisy_img = tf.round(noisy_img)
        noisy_img = tf.clip_by_value(noisy_img,0.,1.)

        # noisy_img = skimage.util.random_noise(input_image, mode="gaussian")
        return noisy_img


    


    def tf_random_crop(self,input_image):
        input_image = tf.image.resize(input_image, [int(self.IMG_SHAPE[0]*1.2), int(self.IMG_SHAPE[0]*1.2)])
        crop_image = tf.image.random_crop(input_image, size=self.IMG_SHAPE)
        
        return crop_image
        

    def tf_flip_aug(self,input_image):
        input_image = tf.reshape(input_image,IMG_SHAPE)
        flipped = tf.image.flip_left_right(input_image)
        flipped = tf.cast(flipped,tf.float32)
        return flipped

    def tf_color_jitter(self,input_image):
        # one can also shuffle the order of following augmentations
        # each time they are applied.
        
        x = input_image
        s=0.7
        x = tf.image.random_brightness(x, max_delta=0.8*s)
        x = tf.image.random_contrast(x, lower=1-0.8*s, upper=1+0.8*s)
        x = tf.image.random_saturation(x, lower=1-0.8*s, upper=1+0.8*s)
        x = tf.image.random_hue(x, max_delta=0.2*s)
        x = tf.clip_by_value(x, 0., 1.)
        
        return x[:,:,0]




    
def to_one_hot(int_label,class_num):


    one_hot_label = tf.one_hot(int_label, depth = class_num)
    
    # return {"data":input["data"],"label":one_hot_label}

    return (one_hot_label)




def make_aug_data_and_dataset(IMG_SHAPE,class_num,target_data,target_label,fn_list,using_contrastive):
    target_dataset = 0
    target_data = target_data.map(lambda x: tf.reshape(x,IMG_SHAPE), tf.data.experimental.AUTOTUNE)
    for i in range(len(fn_list)):
        if i != (len(fn_list) - 1):
            current_start = i
            while current_start + 1 < len(fn_list):
                # for fashionmnist, using this part
                a_x = target_data
                b_x = target_data
                a_x = a_x.map(lambda x: tf.cast(x,tf.float32)/255, tf.data.experimental.AUTOTUNE)
                b_x = b_x.map(lambda x: tf.cast(x,tf.float32)/255, tf.data.experimental.AUTOTUNE)
                a_x = a_x.map(fn_list[i], tf.data.experimental.AUTOTUNE)
                b_x = b_x.map(fn_list[current_start+1], tf.data.experimental.AUTOTUNE)

                label_tmp = target_label
                label_tmp = label_tmp.map(lambda x: to_one_hot(x,class_num), tf.data.experimental.AUTOTUNE)

                x_pair = tf.data.Dataset.zip((a_x,b_x))
                
                if using_contrastive:
                    y_pair = tf.data.Dataset.zip((label_tmp,label_tmp,label_tmp))
                else:
                    y_pair = tf.data.Dataset.zip((label_tmp,label_tmp))

                if target_dataset == 0:
                    target_dataset = tf.data.Dataset.zip((x_pair,y_pair))
                else:
                    target_dataset.concatenate(tf.data.Dataset.zip((x_pair,y_pair)))
                
                current_start+=1
        
    return target_dataset






def compute_contrasive_loss(vec_a, vec_b):
    '''
    vec: (...,c)
    '''
    temp = 0.1
    vec_a = tf.reshape(vec_a,[-1,vec_a.shape[-1]]) # (n,c)
    vec_b = tf.reshape(vec_b,[-1,vec_b.shape[-1]])
    print("vec_a:",vec_a)
    print("vec_b:",vec_b)
    print("type(vec_a.shape[-1]):",type(vec_a.shape[-1]))
    vec_ab = tf.concat([vec_a,vec_b],axis=0) # (2n,c)
    normalized_vec_ab = tf.nn.l2_normalize(vec_ab,axis=-1) # (2n,c)
    pairwise_cos_sim = tf.matmul(normalized_vec_ab,normalized_vec_ab,transpose_b = True)/temp # (2n,2n)
    print("pairwise_cos_sim:",pairwise_cos_sim)


    diagonal_vec = tf.zeros([vec_ab.shape[0]]) # why there's no error????
    diagonal_mat = tf.linalg.diag(diagonal_vec, padding_value=1) # 2n,2n
    
    no_self_matrix = tf.multiply(diagonal_mat,pairwise_cos_sim) # (2n,2n)
    denominator = tf.reduce_sum(tf.math.exp(no_self_matrix))

    # print("no_self_matrix:",no_self_matrix)
    sim_loss = tf.reduce_mean(-1*tf.math.log(tf.math.divide(tf.math.exp(pairwise_cos_sim),denominator)))
    print("sim_loss:",sim_loss)
    return sim_loss

class Contrastive_Loss(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='Contrastive_loss'):
        super(Contrastive_Loss,self).__init__(reduction=reduction, name=name)
    
    def __call__(self,yt,yp,sample_weight=None):
        # print("contra_yt",yt)
        # print("contra_yp",yp)

        # x_embd_a,x_embd_b = yp[0],yp[1]
        single_channel = int(yp.shape[-1]/2)


        x_embd_a,x_embd_b = yp[:,:single_channel],yp[:,single_channel:]

        x_embd_contrasive_loss = compute_contrasive_loss(x_embd_a,x_embd_b) # some problem for gradient passing
        # x_embd_contrasive_loss = tf.keras.losses.cosine_similarity(x_embd_a,x_embd_b)# it works

        return  x_embd_contrasive_loss

class Custom_Cate_Loss(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='Custom_Cate_Loss'):
        super(Custom_Cate_Loss,self).__init__(reduction=reduction, name=name)
    
    def __call__(self,yt,yp,sample_weight=None):

        print("yp",yp.shape)
        print("yt",yt.shape)

        # logits_a,logits_b = yp[0], yp[1]
        # y_a,y_b = yt[0],yt[1]
        # logits_concat = tf.concat([logits_a,logits_b],axis=0)
        # y_concat = tf.concat([y_a,y_b],axis=0)
        cata_loss = tf.keras.losses.categorical_crossentropy(yt,yp, from_logits = True)
        # tf.nn.softmax_cross_entropy_witprint("x_embd_a",x_embd_a)buih_logits(labels, logits, axis=-1, name=None)
        return cata_loss


class Est_Cos_Similarity(tf.keras.metrics.Metric):

    def __init__(self, name='est_similarity', **kwargs):
        super(Est_Cos_Similarity, self).__init__(name=name, **kwargs)

        self.est_cos_similarity = self.add_weight(name='est_similarity', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):

        
        # x_embd_a,x_embd_b = yp[0],yp[1]
        single_channel = int(y_pred.shape[-1]/2)

        x_embd_a,x_embd_b = y_pred[:,:single_channel],y_pred[:,single_channel:]
        cosine_simi = tf.reduce_mean(tf.keras.losses.cosine_similarity(x_embd_a,x_embd_b))

        
        self.est_cos_similarity.assign(cosine_simi)
        

    def result(self):
        return self.est_cos_similarity

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.est_cos_similarity.assign(0.)



if __name__ == "__main__":
    
    



    # define hyperparameters


    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
  
    data_enlarge_scale = 1 # 1920 *n
    epoch_num = 500
    batch_size = 32
    drop_rt = 0.2
    lr = 1e-4
    # filters = 8
    filters = 32
    shuffle_buffer_size = 48000

    # imagenet setting:
    imagenet_csv_path = "../seg_and_depth_estimation/dataset/mini_imagenet/mini-imagenet-tools"
    imagenet_data_path = "../seg_and_depth_estimation/dataset/mini_imagenet/images/"
    IMG_SHAPE = (64,64,3)
    class_num = 100

    # optimizer = tfa.optimizers.LAMB(learning_rate=lr,weight_decay_rate=1e-6)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # # fashion mnist setting
    # IMG_SHAPE = (28,28,1)
    # class_num = 10





    # fn_list = [tf_noise_aug,tf_roll_aug,tf_flip_aug,tf_color_jitter]
    AUG = Augmentation(IMG_SHAPE)
    fn_list = [AUG.tf_noise_aug,AUG.tf_random_crop,AUG.tf_color_jitter]

    
    # tf.compat.v1.disable_eager_execution()

    # define callback and set networ parameters
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    print(gpus)
    tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    if gpus:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


    log_dir="./tf_log/contra"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
    # file_writer.set_as_default()

    save_model_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='./model/Contrastive_Learning_TF2.ckpt',
        save_weights_only= True,
        verbose=1)
    
    contrastive_callback_list = [tensorboard_callback,save_model_callback]


    log_dir="./tf_log/non_contra"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
    # file_writer.set_as_default()

    save_model_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='./model/non_Contrastive_Learning_TF2.ckpt',
        save_weights_only= True,
        verbose=1)
    non_contrastive_callback_list = [tensorboard_callback,save_model_callback]



    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


    '''
    Explore the data
    '''
    # train_images.shape # (60000, 28, 28)
    # len(train_labels) # 60000
    # train_labels # array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)
    # test_images.shape # (10000, 28, 28)
    # len(test_labels) # 10000
    # print(np.max(train_images[0])) # 255


    '''
    For contrastive learning
    '''

    # for fashion mnist:
    # train_dataset = make_aug_data_and_dataset(IMG_SHAPE,class_num,tf.data.Dataset.from_tensor_slices(train_images),\
    # #     tf.data.Dataset.from_tensor_slices(train_labels),fn_list)
    # # test_dataset = make_aug_data_and_dataset(IMG_SHAPE,class_num,tf.data.Dataset.from_tensor_slices(test_images),\
    # #     tf.data.Dataset.from_tensor_slices(test_labels),fn_list)
    
    # # for mini imagenet:
    MID_loader  = MiniImagenetDataset(batch_size,IMG_SHAPE,imagenet_csv_path,imagenet_data_path)
    label_tmp = tf.data.Dataset.from_tensor_slices(MID_loader.train_label_list)
    x_tmp = tf.data.Dataset.from_tensor_slices(MID_loader.train_filename_list)
    x_tmp = x_tmp.map(MID_loader.read_line_and_parse, tf.data.experimental.AUTOTUNE)
    x_tmp = x_tmp.map(MID_loader.crop_and_resize_image, tf.data.experimental.AUTOTUNE)
    val_label_tmp = tf.data.Dataset.from_tensor_slices(MID_loader.val_label_list)
    val_x_tmp = tf.data.Dataset.from_tensor_slices(MID_loader.val_filename_list)
    val_x_tmp = val_x_tmp.map(MID_loader.read_line_and_parse, tf.data.experimental.AUTOTUNE)
    val_x_tmp = val_x_tmp.map(MID_loader.crop_and_resize_image, tf.data.experimental.AUTOTUNE)

    train_dataset = make_aug_data_and_dataset(IMG_SHAPE,class_num,x_tmp,\
        label_tmp,fn_list,using_contrastive=True)
    test_dataset = make_aug_data_and_dataset(IMG_SHAPE,class_num,val_x_tmp,\
        val_label_tmp,fn_list,using_contrastive=True)

    # # for common shuffle and prefetch
    train_dataset = train_dataset.shuffle(buffer_size=shuffle_buffer_size,reshuffle_each_iteration=True).batch(batch_size, drop_remainder=True)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.shuffle(buffer_size=128).batch(batch_size, drop_remainder=True)
    test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)


    # Build Model
    model = Contrastive_Net(IMG_SHAPE,class_num,filters)
    net = model.build(using_contrastive = True)

    
    contrastive_loss_fn = Contrastive_Loss()
    custom_cate_loss_fn = Custom_Cate_Loss()

    metrics = [tf.keras.metrics.CategoricalAccuracy(name='a_accuracy'),\
                tf.keras.metrics.CategoricalAccuracy(name='b_accuracy'),Est_Cos_Similarity()]
    

    
    
    net.compile(optimizer,loss = [keras.losses.CategoricalCrossentropy(from_logits=True),\
        keras.losses.CategoricalCrossentropy(from_logits=True),\
            contrastive_loss_fn],loss_weights=[1., 1.,1.],\
            metrics=metrics)



    # net.summary()


    # # This verify that the variables are reused correctly.

    # for var in net.variables:
    #     print(var.name)

   
    net.fit(x = train_dataset,epochs=epoch_num,validation_data=test_dataset,validation_freq = 5,callbacks = contrastive_callback_list)



    '''
    for non contrastive learning
    '''
#     # # for fashion mnist:
#     # train_dataset = make_aug_data_and_dataset(IMG_SHAPE,class_num,tf.data.Dataset.from_tensor_slices(train_images),\
#     #     tf.data.Dataset.from_tensor_slices(train_labels),fn_list)
#     # test_dataset = make_aug_data_and_dataset(IMG_SHAPE,class_num,tf.data.Dataset.from_tensor_slices(test_images),\
#     #     tf.data.Dataset.from_tensor_slices(test_labels),fn_list)
    
#     # for mini imagenet:
#     MID_loader  = MiniImagenetDataset(batch_size,IMG_SHAPE,imagenet_csv_path,imagenet_data_path)
#     label_tmp = tf.data.Dataset.from_tensor_slices(MID_loader.train_label_list)
#     x_tmp = tf.data.Dataset.from_tensor_slices(MID_loader.train_filename_list)
#     x_tmp = x_tmp.map(MID_loader.read_line_and_parse, tf.data.experimental.AUTOTUNE)
#     x_tmp = x_tmp.map(MID_loader.crop_and_resize_image, tf.data.experimental.AUTOTUNE)
#     val_label_tmp = tf.data.Dataset.from_tensor_slices(MID_loader.val_label_list)
#     val_x_tmp = tf.data.Dataset.from_tensor_slices(MID_loader.val_filename_list)
#     val_x_tmp = val_x_tmp.map(MID_loader.read_line_and_parse, tf.data.experimental.AUTOTUNE)
#     val_x_tmp = val_x_tmp.map(MID_loader.crop_and_resize_image, tf.data.experimental.AUTOTUNE)

#     train_dataset = make_aug_data_and_dataset(IMG_SHAPE,class_num,x_tmp,\
#         label_tmp,fn_list,using_contrastive=False)
#     test_dataset = make_aug_data_and_dataset(IMG_SHAPE,class_num,val_x_tmp,\
#         val_label_tmp,fn_list,using_contrastive=False)

#     # for common shuffle and prefetch
#     train_dataset = train_dataset.shuffle(buffer_size=shuffle_buffer_size,reshuffle_each_iteration=True).batch(batch_size, drop_remainder=True)
#     train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
#     test_dataset = test_dataset.shuffle(buffer_size=128).batch(batch_size, drop_remainder=True)
#     test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)
#     # Build Model
#     model = Contrastive_Net(IMG_SHAPE,class_num,filters)
#     net = model.build(using_contrastive = False)

#     net.summary()

#     # for var in net.variables:
#     #     print(var.name)

#     custom_cate_loss_fn = Custom_Cate_Loss()

#     metrics = [tf.keras.metrics.CategoricalAccuracy(name='a_accuracy'),\
#                 tf.keras.metrics.CategoricalAccuracy(name='b_accuracy')]

#     net.compile(optimizer,loss = {"model":keras.losses.CategoricalCrossentropy(from_logits=True),\
#         "model_1":keras.losses.CategoricalCrossentropy(from_logits=True)\
#             },loss_weights=[1., 1.],\
#             metrics=metrics)

#     net.fit(x = train_dataset,epochs=epoch_num,validation_data=test_dataset,validation_freq = 5,callbacks = non_contrastive_callback_list)
# #
# #
# #
# #
# #
#    
#
#
#
# #
