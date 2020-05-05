import tensorflow as tf
import tensorflow_addons as tfa
from net.contrastive_net import Contrastive_Net
from tensorflow import keras
import matplotlib.pyplot as plt

import os

import numpy as np
import argparse


def noise_aug(input_image):
    SNR = 0.8
    max_signal = np.max(input_image) # as 1
    input_shape = [28,28]

    mu, sigma = 0, 0.1
    noise_vector = np.random.normal(mu, sigma, input_shape[0]*input_shape[1])
    noise_vector = noise_vector * (max_signal)*(SNR**2)
    noise_mat = noise_vector.reshape(input_shape)

    noisy_img = input_image + noise_mat
    noisy_img = noisy_img.astype(int)
    noisy_img[noisy_img>=255] = 255
    noisy_img[noisy_img<0] = 0

    # noisy_img = skimage.util.random_noise(input_image, mode="gaussian")
    return noisy_img

def tf_noise_aug(input_image):
    SNR = 0.8
    max_signal = tf.reduce_max(input_image) # as 1
    input_shape = [28,28]

    mu, sigma = 0.0, 0.1
    noise_vector = tf.random.normal([input_shape[0]*input_shape[1]],mu,sigma)
    noise_vector = noise_vector * (tf.cast(max_signal,dtype=tf.float32))*(SNR**2)
    noise_mat = tf.reshape(noise_vector,input_shape)

    noisy_img = tf.cast(input_image,dtype=tf.float32) + noise_mat
    noisy_img = tf.round(noisy_img)
    noisy_img = tf.clip_by_value(noisy_img,0.,255.)

    # noisy_img = skimage.util.random_noise(input_image, mode="gaussian")
    return noisy_img

def tf_roll_aug(input_image):
    x_shift = tf.random.uniform([1],minval=-2,maxval=3,dtype=tf.int32)
    y_shift = tf.random.uniform([1],minval=-2,maxval=3,dtype=tf.int32)

    shiifted_img = tf.roll(input_image,[y_shift[0],x_shift[0]],axis=[0,1])
    shiifted_img = tf.cast(shiifted_img,tf.float32)

    return shiifted_img

def tf_random_crop(input_image):
    # in paper it says crop of random size is (uniform from 0.08 to 1.0 in area)
    # and random aspect ratio(h,w) 3/4 to 4/3
    input_image = tf.expand_dims(input_image,axis=-1)
    
    bounding_boxes = np.array([[[0,0,1,1]]])
    CROP_SIZE= np.array([28,28])
    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
            tf.shape(input_image),
            bounding_boxes=bounding_boxes,
            min_object_covered=0.5)
    bbox_for_draw = tf.reshape(bbox_for_draw,[1,4])
    crop_image = tf.image.crop_and_resize(tf.expand_dims(input_image,axis=0), bbox_for_draw, [0], CROP_SIZE)
    # Random flipping
    do_flip = tf.random.uniform([], 0, 1)
    crop_image = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(crop_image), lambda: crop_image)
    crop_image = tf.reshape(crop_image,[28,28])
    
    return crop_image

def tf_flip_aug(input_image):
    input_image = tf.reshape(input_image,[28,28,1])
    flipped = tf.image.flip_left_right(input_image)
    flipped = tf.reshape(flipped,[28,28])
    flipped = tf.cast(flipped,tf.float32)
    return flipped

def tf_color_jitter(input_image):
    # one can also shuffle the order of following augmentations
    # each time they are applied.
    
    s=0.7
    # x = tf.reshape(input_image,[28,28,1])
    x = tf.stack([input_image,input_image,input_image],axis=-1)
    x = tf.cast(x,dtype=tf.float32)/255
    x = tf.image.random_brightness(x, max_delta=0.8*s)
    x = tf.image.random_contrast(x, lower=1-0.8*s, upper=1+0.8*s)
    x = tf.image.random_saturation(x, lower=1-0.8*s, upper=1+0.8*s)
    x = tf.image.random_hue(x, max_delta=0.2*s)
    x = tf.round(x*255)
    x = tf.clip_by_value(x, 0., 255.)
    

    return x[:,:,0]


def get_data_aug(input_data,fn_list):
    '''
    input_data: (60000, 28, 28)
    fn_list:[tf_noise_aug,tf_roll_aug,tf_flip_aug,tf_color_jitter]
    '''

    input_holder = tf.compat.v1.placeholder(tf.int32,[None,28,28],name="input_holder")
    data_op_list = []
    data_op_list.append(input_holder*1)
    
    
    for i in range(len(fn_list)):
        if i != (len(fn_list) - 1):
            print("fn_list[i]:",fn_list[i])
            aug_op_out = tf.map_fn(fn_list[i],input_holder,parallel_iterations = 10)
      
            data_op_list.append(aug_op_out)
    
    
    data_list = []
    with tf.compat.v1.Session() as sess:

        for output_op in data_op_list:
            # print(output_op)
            op_output = sess.run(output_op,feed_dict={input_holder:input_data.astype(np.int32)})
            print(type(op_output))
            data_list.append(op_output)
        
        sess.close()
    
    tf.compat.v1.reset_default_graph()
    return data_list

    # data_pair_list = []
    # for i in range(len(data_list)):
    #     if i + 1 <= len(data_list):
    #         residue = data_list[i+1:]
    #         for j in range(len(residue)):
    #             data_pair_list.append([data_list[i],residue[j]])
    
    # return data_pair_list



def make_aug_dataset(target_data_list, target_label):

        target_dataset = 0
        for i in range(len(target_data_list)):
            current_start = i
            while current_start + 1 < len(target_data_list):
                a_x = tf.data.Dataset.from_tensor_slices(target_data_list[current_start])
                b_x = tf.data.Dataset.from_tensor_slices(target_data_list[current_start+1])
                

                label_tmp = tf.data.Dataset.from_tensor_slices(target_label)
                label_tmp = label_tmp.map(to_one_hot)

                x_pair = tf.data.Dataset.zip((a_x,b_x))
                y_pair = tf.data.Dataset.zip((label_tmp,label_tmp))
                
                if target_dataset == 0:
                    target_dataset = tf.data.Dataset.zip((x_pair,y_pair))
                else:
                    target_dataset.concatenate(tf.data.Dataset.zip((x_pair,y_pair)))
                current_start+=1
                print("current_start:",current_start)
        
        return target_dataset
    
def to_one_hot(int_label):


    one_hot_label = tf.one_hot(int_label, depth = 10)
    
    # return {"data":input["data"],"label":one_hot_label}

    return (one_hot_label)




def make_aug_data_and_dataset(target_data,target_label,fn_list):
    target_dataset = 0
    for i in range(len(fn_list)):
        if i != (len(fn_list) - 1):
            current_start = i
            while current_start + 1 < len(fn_list):
                print("fn_list[i]:",fn_list[i])
                # aug_op_out = tf.map_fn(fn_list[i],input_holder,parallel_iterations = 10)
                a_x = tf.data.Dataset.from_tensor_slices(target_data)
                b_x = tf.data.Dataset.from_tensor_slices(target_data)
                a_x = a_x.map(fn_list[i])
                b_x = b_x.map(fn_list[current_start+1])

                label_tmp = tf.data.Dataset.from_tensor_slices(target_label)
                label_tmp = label_tmp.map(to_one_hot)

                x_pair = tf.data.Dataset.zip((a_x,b_x))
                y_pair = tf.data.Dataset.zip((label_tmp,label_tmp))

                if target_dataset == 0:
                    target_dataset = tf.data.Dataset.zip((x_pair,y_pair))
                else:
                    target_dataset.concatenate(tf.data.Dataset.zip((x_pair,y_pair)))
                
                current_start+=1
        
    return target_dataset


if __name__ == "__main__":
    
    



    # define hyperparameters


    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
  

    data_enlarge_scale = 1 # 1920 *n

    epoch_num = 500
    batch_size = 3200
    
    class_num = 10
    drop_rt = 0.2
    lr = 1e-4

     
    filters = 8

    # fn_list = [tf_noise_aug,tf_roll_aug,tf_flip_aug,tf_color_jitter]
    fn_list = [tf_noise_aug,tf_random_crop,tf_color_jitter]

    
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
    # log_dir="./tf_log/non_contra"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
    file_writer.set_as_default()

    save_model_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='./model/Contrastive_Learning_TF2.ckpt',
        # Path where to save the model
        # The two parameters below mean that we will overwrite
        # the current checkpoint if and only if
        # the `val_loss` score has improved.
        save_best_only=True,
        # monitor='val_acc',
        save_weights_only= True,
        monitor='val_categorical_accuracy',
        verbose=1)
    

    callback_list = [tensorboard_callback,save_model_callback]



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
    Do Augmentation via tf.session...feel so gooooooood!!!!!

    But it is incompatible with fit 
    '''
    # train_data_list = get_data_aug(train_images,fn_list)
    # test_data_list = get_data_aug(test_images,fn_list)


    # # plt.figure(figsize=(10,10))
    # # for i in range(25):
    # #     plt.subplot(5,5,i+1)
    # #     plt.xticks([])
    # #     plt.yticks([])
    # #     plt.grid(False)
    # #     # plt.imshow(noise_aug(train_images[i]), cmap=plt.cm.binary)
    # #     # plt.imshow(tf_flip_aug(train_images[i]), cmap=plt.cm.binary)
    # #     # plt.imshow(tf_shift_aug(train_images[i]), cmap=plt.cm.binary)
        
    # #     plt.imshow(train_data_list[3][i], cmap=plt.cm.binary)
    # #     plt.xlabel(class_names[train_labels[i]])
    # # plt.savefig("./gen_img/test_tf_jitter.png")
    # # # plt.show()

    # train_dataset = make_aug_dataset(train_data_list,train_labels)
    # train_dataset = train_dataset.shuffle(buffer_size=128,reshuffle_each_iteration=True).batch(batch_size, drop_remainder=True)
    # train_dataset = train_dataset.prefetch(batch_size)

    # test_dataset = make_aug_dataset(test_data_list,test_labels)   
    # test_dataset = test_dataset.shuffle(buffer_size=128).batch(batch_size, drop_remainder=True)
    # test_dataset = test_dataset.prefetch(batch_size)

    '''
    Do augmentation and build model by functional API 
    '''

    train_dataset = make_aug_data_and_dataset(train_images,train_labels,fn_list)
    train_dataset = train_dataset.shuffle(buffer_size=128,reshuffle_each_iteration=True).batch(batch_size, drop_remainder=True)
    train_dataset = train_dataset.prefetch(batch_size)

    test_dataset = make_aug_data_and_dataset(test_images,test_labels,fn_list)
    test_dataset = test_dataset.shuffle(buffer_size=128).batch(batch_size, drop_remainder=True)
    test_dataset = test_dataset.prefetch(batch_size)

    # Build Model
    model = Contrastive_Net((28,28),class_num,filters)
    net = model.build()
    net.add_loss(model.custom_loss()) 

    # optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    optimizer = tfa.optimizers.LAMB(learning_rate=lr,weight_decay_rate=1e-6)
    net.compile(optimizer,loss = lambda yt, yp: tf.keras.losses.categorical_crossentropy(yt, yp, from_logits = True),metrics= [tf.keras.metrics.CategoricalAccuracy()])
    net.summary()


    # This verify that the variables are reused correctly.
    for var in net.variables:
        print(var.name)


    net.fit(x = train_dataset,epochs=epoch_num,validation_data=test_dataset,validation_freq = 5,callbacks = callback_list)
    # net.fit(x = train_dataset,epochs=epoch_num)





    




