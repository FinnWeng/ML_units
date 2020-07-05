import tensorflow as tf
import os
import time
import datetime
import sys
from PIL import Image
import numpy as np

import random
import tensorflow_addons as tfa
import csv
import cv2
import random



class MiniImagenetDataset:
    def __init__(self,batch_size,IMG_SHAPE,csv_path,data_path):
        self.batch_size = batch_size
        # encoder resnet 101 (origin v1 and resnext, I like to use v2 or resnext) 
        # self.data_path = data_path # for bts's dataset
        # data_path = './dataset/' # for NYU Origin dataset
        # self.gen_img_path = gen_img_path

        # self.filenames_file = filenames_file
        # self.degree = 5

        # the true target size
        # input_height = 416
        # input_width = 544

        self.csv_path = csv_path
        self.data_path = data_path

        self.split_lists = ['train', 'val', 'test'] # 64,16,20 categories
        self.csv_files = ['/csv_files/train.csv','/csv_files/val.csv', '/csv_files/test.csv']
        filename_list = []
        label_list = []

        
        self.IMG_SHAPE =  IMG_SHAPE

        
        for this_split in self.split_lists: # all 100 classes, val split by fit function
            filename = self.csv_path + '/csv_files/' + this_split + '.csv'
            with open(filename) as csvfile:
                csv_reader = csv.DictReader(csvfile, delimiter=',')
                current_label = -1
                current_label_num = -1
                for row in csv_reader:
                    if current_label != row["label"]:
                        current_label = row["label"]
                        current_label_num +=1
                    
                    filename_list.append(row["filename"])
                    label_list.append(current_label_num)



        self.filename_list = filename_list
        self.label_list = label_list 

        x_y_pair_list = list(zip(self.filename_list,self.label_list))
        

        self.train_filename_list = []
        self.val_filename_list = []
        self.train_label_list = []
        self.val_label_list = []

        
        for i in range(100): # 100 class, each 600 sample
            sub_pair = x_y_pair_list[i*600:(i+1)*600]

            random.shuffle(sub_pair)
            
            
            train_pair = list(zip(*sub_pair[:480]))
            val_pair = list(zip(*sub_pair[480:]))
            
            print("val_pair:",val_pair)
            self.train_filename_list.extend(train_pair[0])
            self.val_filename_list.extend(val_pair[0])
            self.train_label_list.extend(train_pair[1])
            self.val_label_list.extend(val_pair[1])
        # print("train_label_list:",self.train_label_list)

        self.number_of_class = len(list(set(label_list)))
        self.dataset_size = len(label_list)
    
    def read_line_and_parse(self,filename):
        
        image_path = tf.strings.join([self.data_path, filename])
       
        image = tf.image.decode_jpeg(tf.io.read_file(image_path))

        # image = tf.image.convert_image_dtype(image, tf.float32) # 255-0 => 0-1

        return image
    
    def crop_and_resize_image(self,image):
        image = tf.cast(image,tf.float32)
        target_size = self.IMG_SHAPE[0]*self.IMG_SHAPE[1]
        target_size = tf.cast(target_size,tf.float32)
        image_shape = tf.shape(image)
        image_shape = tf.cast(image_shape,tf.float32)
        image_size = image_shape[0]*image_shape[1]
        enlarge_fn = lambda: tf.image.resize(image,(tf.cast(image_shape[0]*1.5,tf.int32),\
            tf.cast(image_shape[1]*1.5,tf.int32)))
        shrink_fn = lambda: tf.image.resize(image,(tf.cast(image_shape[0]/1.5,tf.int32),tf.cast(image_shape[1]/1.5,tf.int32)))
        image = tf.cond(tf.cast(target_size,tf.int32) > tf.cast(image_size*1.5,tf.int32) , enlarge_fn,lambda: image) # if not large enough, enlarge
        image = tf.cond(tf.cast(target_size*1.5,tf.int32) < tf.cast(image_size,tf.int32) , shrink_fn,lambda: image ) # if too large, shrink

        image = tf.image.resize_with_crop_or_pad(image,self.IMG_SHAPE[0],self.IMG_SHAPE[1])
        # image = tf.cast(image*255,tf.int32)
        return image

    
    def make_dataset(self):

        label_tmp = tf.data.Dataset.from_tensor_slices(self.label_list)
        label_tmp = label_tmp.map(self.to_one_hot)

        x_tmp = tf.data.Dataset.from_tensor_slices(self.filename_list)
        x_tmp = x_tmp.map(self.read_line_and_parse, tf.data.experimental.AUTOTUNE)
        x_tmp = x_tmp.map(self.crop_and_resize_image, tf.data.experimental.AUTOTUNE)

        loader = tf.data.Dataset.zip((x_tmp,label_tmp))
        loader = loader.shuffle(len(self.label_list)//10).repeat()
        loader = loader.batch(self.batch_size)
        loader = loader.prefetch(tf.data.experimental.AUTOTUNE)
        return loader
    

    def to_one_hot(self,int_label):


        one_hot_label = tf.one_hot(int_label, depth = self.number_of_class)
        
        # return {"data":input["data"],"label":one_hot_label}

        return (one_hot_label)
    
    
    
    def __call__(self):
        

        # return self.label_list
        return self.filename_list



if __name__ == '__main__':
    csv_path = "../../dataset/mini_imagenet/mini-imagenet-tools"
    data_path = "../../dataset/mini_imagenet/images/"
    IMG_SHAPE = (288,512, 3)

    MID_loader = MiniImagenetDataset(10,IMG_SHAPE,csv_path,data_path)
    dataset = MID_loader.make_dataset()

    test_gen_img_path = "../test_gen_img/"
    count = 0
    for image,label in dataset:
        print(image.shape,label.shape)
        # cv2.imwrite("../test_gen_img/"+ str(count)+".jpg",(image.numpy()[0]*255).astype(np.uint8))
        count+=1
        print(count)

