#!/usr/bin/python
#coding=utf-8

import tensorflow as tf
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def read_cifar10(data_dir, is_train, batch_size, shuffle):
    """
    Read cifar10 data
    :param data_dir: data directory
    :param is_train: input train data or test data
    :param batch_size: batch size
    :param shuffle: whether shuffle the data
    :return: label: 1D tensor, [batch_size, n_classes], one-hot coding, tf.int32
             images: 4D tensor, [batch_size, width, height, 3], tf.float32
    """

    img_width = 32
    img_height = 32
    img_channel = 3
    label_bytes = 1
    image_bytes = img_width * img_height * img_channel

    with tf.name_scope('input'):

        data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')

        if is_train:
            filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % ii) for ii in np.arange(1, 6)]
        else:
            filenames = [os.path.join(data_dir, 'test_batch.bin')]

        filename_queue = tf.train.input_producer(filenames)
        reader = tf.FixedLengthRecordReader(label_bytes + image_bytes)
        key, value = reader.read(filename_queue)
        record_bytes = tf.decode_raw(value, tf.uint8)

        label = tf.slice(record_bytes, [0], [label_bytes])
        label = tf.cast(label, tf.int32)

        image_raw = tf.slice(record_bytes, [label_bytes], [image_bytes])
        image_raw = tf.reshape(image_raw, [img_channel, img_height, img_width])
        image = tf.transpose(image_raw, (1, 2, 0))  # convert D/H/W -> H/W/D
        image = tf.cast(image, tf.float32)

        # normalization: (x - mean) / var
        image = tf.image.per_image_standardization(image)

        # tf.train.shuffle_batch() Args:
        #
        # tensors: The list or dictionary of tensors to enqueue.
        # batch_size: The new batch size pulled from the queue.
        # capacity: An integer. The maximum number of elements in the queue.
        # min_after_dequeue: Minimum number elements in the queue after a dequeue,
        #                    used to ensure a level of mixing of elements.
        # num_threads: The number of threads enqueuing tensor_list.
        if shuffle:
            images, label_batch = tf.train.shuffle_batch([image, label],
                                                         batch_size=batch_size,
                                                         capacity=20000,
                                                         min_after_dequeue=3000,
                                                         num_threads=64)
        else:
            images, label_batch = tf.train.batch([image, label],
                                                 batch_size=batch_size,
                                                 capacity=2000,
                                                 num_threads=64)
        # one-hot coding
        n_classes = 10
        label_batch = tf.one_hot(label_batch, depth=n_classes)
        label_batch = tf.cast(label_batch, dtype=tf.int32)
        label_batch = tf.reshape(label_batch, [batch_size, n_classes])

        return images, label_batch

def read_armyData(file_list, is_train = True, batch_size=32, shuffle=False):
    """
    Read ArmyData
    :param file_list: a text containing names of images,
        such as :[['./dataset/deq.jpg',4],['./dataset/qwe.jpg', 6]]
    :param is_train: input train data or test data
    :param batch_size: batch size
    :param shuffle: whether shuffle the data
    :return: label: 1D tensor, [batch_size, n_classes], one-hot coding, tf.int32
             images: 4D tensor, [batch_size, width, height, 3], tf.float32
    """
    img_width = 3500
    img_height = 2400
    img_channel = 3
    names = np.loadtxt(file_list, dtype=bytes,unpack = True, usecols = (0)).astype(str)
    #get file name
    namelist = []
    for name in names:
        name = os.path.join(config.dataPath,name)
        namelist.append(name)
    labels = np.loadtxt(file_list,unpack = True,usecols = (1))
    
    imgName = tf.cast(namelist, tf.string)
    labels = tf.cast(labels,tf.int32)
    input_queue = tf.train.slice_input_producer([imgName, labels])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents,channels = 3)
   
   #need to focus, this is where to consider
    image = tf.image.resize_images(image, (img_height,img_width))
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    if shuffle:
        images, label_batch = tf.train.shuffle_batch([image, label],
                                                     batch_size=batch_size,
                                                     capacity=20000,
                                                     min_after_dequeue=3000,
                                                     num_threads=64)
    else:
        images, label_batch = tf.train.batch([image, label],
                                             batch_size=batch_size,
                                             capacity=2000,
                                             num_threads=64)
    # one-hot coding
    n_classes = config.N_CLASSES
    label_batch = tf.one_hot(label_batch, depth=n_classes)
    label_batch = tf.cast(label_batch, dtype=tf.int32)
    label_batch = tf.reshape(label_batch, [batch_size, n_classes])

    return images, label_batch

def read_lvData(file_csv, is_train=True, batch_size=32, shuffle=False):
    """
    Read lvData 
    :param file_csv: a csv containing names of images and labels
        such as :['./dataset/deq.jpg',4
                  './dataset/qwe.jpg', 6]
    :param is_train: input train data or test data
    :param batch_size: batch size
    :param shuffle: whether shuffle the data
    :return: label: 1D tensor, [batch_size, n_classes], one-hot coding, tf.int32
             images: 4D tensor, [batch_size, width, height, 3], tf.float32

                                label
    正常	norm              0 
    不导电	defect1           1
    擦花	defect2           2
    横条压凹	defect3           3
    桔皮	defect4           4
    漏底	defect5           5
    碰伤	defect6           6
    起坑	defect7           7
    凸粉	defect8           8
    涂层开裂	defect9           9
    脏点	defect10          10
    其他	defect11          11
    """

    #dataset info
    img_width = 256
    img_height = 192
    img_channel = 3
    N_CLASSES = 12

    data = pd.read_csv(file_csv)
    #stratify 分层抽样e
    train_x, val_x, train_y, val_y = train_test_split(data['img_path'], data['label'], test_size=0.3, random_state=666, stratify=data['label'])
    if is_train:
        imgName = tf.cast(train_x, tf.string)
        labels = tf.cast(train_y, tf.int32)
    else:
        imgName = tf.cast(val_x, tf.string)
        labels = tf.cast(val_y, tf.int32)
    input_queue = tf.train.slice_input_producer([imgName, labels])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents,channels = 3)
   
    #need to focus, this is where to consider
    image = tf.image.resize_images(image, (img_height,img_width))
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    if shuffle:
        images, label_batch = tf.train.shuffle_batch([image, label],
                                                     batch_size=batch_size,
                                                     capacity=20000,
                                                     min_after_dequeue=3000,
                                                     num_threads=64)
    else:
        images, label_batch = tf.train.batch([image, label],
                                             batch_size=batch_size,
                                             capacity=2000,
                                             num_threads=64)
    # one-hot coding
    label_batch = tf.one_hot(label_batch, depth=N_CLASSES)
    label_batch = tf.cast(label_batch, dtype=tf.int32)
    label_batch = tf.reshape(label_batch, [batch_size, N_CLASSES])

    return images, label_batch


if __name__=='__main__':
    images, label_batch = read_cifar10('./cifar10',True,32,True)
    sess = tf.Session()
    coord = tf.train.Coordinator();
    threads = tf.train.start_queue_runners(sess, coord = coord)
    label_batch = sess.run(label_batch)
    print(label_batch)
    
    #images, label_batch = read_armyData('./dataset/armydata/filelist.txt',True, batch_size = 32, shuffle = False)
    #sess = tf.Session()
    #coord = tf.train.Coordinator();
    #threads = tf.train.start_queue_runners(sess, coord = coord)
    #label_batch = sess.run(label_batch)
    #print(label_batch)
    
    #images_op, labels_op = read_lvData('./data/label.csv')
    #sess = tf.Session()
    #coord = tf.train.Coordinator();
    #threads = tf.train.start_queue_runners(sess, coord = coord)
    #images, labels = sess.run([images_op, labels_op])
    #print(images)

