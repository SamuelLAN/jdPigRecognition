#!/usr/bin/Python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import random
import zipfile
import numpy as np
from PIL import Image
from six.moves.urllib.request import urlretrieve
import threading

import tensorflow as tf

if '2.7' in sys.version:
    import Queue as queue
else:
    import queue
import time


class Data:
    DATA_ROOT = r'../data/TrainImgMore'
    RESIZE = [224, 224]
    # RESIZE = [39, 39]
    RATIO = 1.0
    NUM_CLASSES = 21

    def __init__(self, file_path, img_dir, label_dir, start_ratio=0.0, end_ratio=1.0, name='', resize=None):
        self.__chang_dir()

        self.__file_path = file_path
        self.__img_dir = img_dir
        self.__label_dir = label_dir

        # 初始化变量
        self.__name = name
        self.__data = []
        self.__resize = resize if resize else self.RESIZE

        # 加载全部数据
        self.__load()
        self.__data_len = len(self.__data)

        # 检查输入参数
        start_ratio = min(max(0.0, start_ratio), 1.0)
        end_ratio = min(max(0.0, end_ratio), 1.0)

        # 根据比例计算数据的位置范围
        start_index = int(self.__data_len * start_ratio)
        end_index = int(self.__data_len * end_ratio)

        # 根据数据的位置范围 取数据
        self.__data = self.__data[start_index: end_index]

        self.__data_len = len(self.__data)
        random.shuffle(self.__data)

        self.__queue = queue.Queue()
        self.__stop_thread = False
        self.__thread = None

        self.__cur_index = 0

        self.__sess = tf.Session()

        self.__label_ph = tf.placeholder(tf.int32, [None, None])
        self.__one_hot = tf.one_hot(self.__label_ph, self.NUM_CLASSES)

        self.__sess.run(tf.global_variables_initializer())

    @staticmethod
    def __chang_dir():
        # 将运行路径切换到当前文件所在路径
        cur_dir_path = os.path.split(__file__)[0]
        if cur_dir_path and os.path.abspath(os.path.curdir) != os.path.abspath(cur_dir_path):
            os.chdir(cur_dir_path)
            sys.path.append(cur_dir_path)

    ''' 加载数据 '''

    def __load(self):
        self.echo('Loading %s data ...' % self.__name)

        with open(self.__file_path, 'rb') as f:
            file_list = f.read().decode('utf-8').split(u'\n')

        file_len = len(file_list)

        for i, file_name in enumerate(file_list):
            progress = float(i + 1) / file_len * 100
            self.echo('\r >> progress: %.2f%% \t' % progress, False)

            if not file_name:
                continue
            self.__data.append(file_name)

        self.echo('\nFinish Loading\n')

    def __get_data(self):
        max_q_size = min(self.__data_len, 500)
        while not self.__stop_thread:
            while self.__queue.qsize() <= max_q_size:
                file_name = self.__data[self.__cur_index]
                x, y = self.__get_x_y(file_name)

                self.__queue.put([x, y])
                self.__cur_index = (self.__cur_index + 1) % self.__data_len

            time.sleep(0.3)

        self.echo(
            '\n*************************************\n Thread "get_%s_data" stop\n***********************\n' % self.__name)

    def start_thread(self):
        self.__thread = threading.Thread(target=self.__get_data, name=('get_%s_data' % self.__name))
        self.__thread.start()
        self.echo('Thread "get_%s_data" is running ... ' % self.__name)

    def stop(self):
        self.__stop_thread = True

    def __get_x_y(self, file_name):
        img_path = os.path.join(self.__img_dir, file_name + u'.jpg')
        label_path = os.path.join(self.__label_dir, file_name + u'.png')

        img = self.add_padding(img_path)
        label = self.add_padding(label_path)

        label[label == 255] = 0

        label = self.__sess.run(self.__one_hot, feed_dict={self.__label_ph: label})
        return img, label

    def __resize_np_img(self, np_image):
        return np.array(Image.fromarray(np_image).resize(self.__resize), dtype=np.float32)

    def add_padding(self, img_path):
        image = Image.open(img_path)
        w, h = image.size
        ratio = float(w) / h

        if abs(ratio - Data.RATIO) <= 0.1:
            return np.array(image.resize(self.__resize))

        np_image = np.array(image)
        has_expand = False
        if len(np_image.shape) == 2:
            has_expand = True
            np_image = np.expand_dims(np_image, axis=3)
        h, w, c = np_image.shape

        if ratio > Data.RATIO:
            new_h = int(float(w) / Data.RATIO)
            padding = int((new_h - h) / 2.0)

            np_new_image = np.zeros([new_h, w, c])
            np_new_image[padding: padding + h, :, :] = np_image

        else:
            new_w = int(float(h) * Data.RATIO)
            padding = int((new_w - w) / 2.0)

            np_new_image = np.zeros([h, new_w, c])
            np_new_image[:, padding: padding + w, :] = np_image

        tmp_np_new_image = np.cast['uint8'](np_new_image)
        if has_expand:
            tmp_np_new_image = np.squeeze(tmp_np_new_image, axis=-1)

        new_image = Image.fromarray(tmp_np_new_image)
        return np.array(new_image.resize(self.__resize))

    def next_batch(self, batch_size):
        X = []
        y = []
        for i in range(batch_size):
            while self.__queue.empty():
                time.sleep(0.2)
            if not self.__queue.empty():
                _x, _y = self.__queue.get()
                X.append(_x)
                y.append(_y)
        return np.array(X), np.array(y)

    ''' 获取数据集大小 '''

    def get_size(self):
        return self.__data_len

    # ''' 重置当前 index 位置 '''
    # def reset_cur_index(self):
    #     self.__cur_index = 0

    ''' 输出展示 '''

    @staticmethod
    def echo(msg, crlf=True):
        if crlf:
            print(msg)
        else:
            try:
                sys.stdout.write(msg)
                sys.stdout.flush()
            except:
                print(msg)


# o_train_data = Data(r'/Users/yusenlin/Documents/github/unet_tf/data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt',
#                     r'/Users/yusenlin/Documents/github/unet_tf/data/VOCdevkit/VOC2012/JPEGImages',
#                     r'/Users/yusenlin/Documents/github/unet_tf/data/VOCdevkit/VOC2012/SegmentationClass',
#                     0.0, 1.0, 'train')
#
# print('size:')
# print(o_train_data.get_size())
#
# o_train_data.start_thread()
#
# import matplotlib.pyplot as plt
#
# for i in range(10):
#     batch_x, batch_y = o_train_data.next_batch(10)
#
#     print('\n*************** %d *****************' % i)
#     print(o_train_data.get_size())
#     print(batch_x.shape)
#     print(batch_y.shape)
#
#     tmp_x = batch_x[0]
#     tmp_y = batch_y[0]
#
#     # o_tmp = Image.fromarray(tmp_x)
#     # o_tmp.show()
#
#     # plt.imshow(tmp_y)
#     # plt.show()
#
#     time.sleep(1)
#
# o_train_data.stop()

# print 'y 0:'
# print batch_y[0]
#
# tmp_x_list = batch_x_list[0]
#
# print 'tmp_x_list'
# for i, x in enumerate(tmp_x_list):
#     print i
#     print x.shape
#     # o_tmp = Image.fromarray(x)
#     # o_tmp.show()
