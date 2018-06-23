#!/usr/bin/Python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import copy
import random
import numpy as np
from PIL import Image


'''
 Data: 取数据到基类
 对外提供接口:
    get_size()
    next_batch()
'''


class Data:
    CUR_DIR = os.path.abspath(os.path.split(__file__)[0])
    PRJ_DIR = os.path.split(CUR_DIR)[0]
    DATA_ROOT = os.path.join(PRJ_DIR, 'data', 'cells')
    IMAGE_SCALE = 2
    RESIZE_SIZE = [360, 360]

    def __init__(self, start_ratio=0.0, end_ratio=1.0, name='', sort_list=[]):
        # 初始化变量
        self.__name = name
        self.__data = []
        self.__data_dict = {}
        self.__data_list = []
        self.__y = {}
        self.__total_size = 0

        self.__sort_list = sort_list

        # 加载全部数据
        self.__load()
        self.__data_len = len(self.__data_list)

        # 检查输入参数
        start_ratio = min(max(0.0, start_ratio), 1.0)
        end_ratio = min(max(0.0, end_ratio), 1.0)

        # 根据比例计算数据的位置范围
        start_index = int(self.__data_len * start_ratio)
        end_index = int(self.__data_len * end_ratio)

        # 根据数据的位置范围 取数据
        self.__data_list = self.__data_list[start_index: end_index]

        for (img_no, data_list) in self.__data_list:
            for data in data_list:
                self.__data.append(data)

        self.__data_len = len(self.__data)
        random.shuffle(self.__data)

        self.__cur_index = 0

    ''' 加载数据 '''

    def __load(self):
        self.echo('Loading %s data ...' % self.__name)
        file_list = os.listdir(self.DATA_ROOT)
        file_len = len(file_list)

        for i, file_name in enumerate(file_list):
            progress = float(i) / file_len * 100
            self.echo('\rprogress: %.2f%% \t' % progress, False)

            if os.path.splitext(file_name)[1].lower() != '.png':
                continue

            if 'mask' in file_name and file_name not in self.__y:
                self.__y[file_name] = self.__get_mask(file_name)

            if 'mask' not in file_name:
                img_no = os.path.splitext(file_name)[0]
                y_file_name = img_no + '_mask.png'

                if y_file_name not in self.__y:
                    self.__y[y_file_name] = self.__get_mask(y_file_name)

                image = Image.open(os.path.join(self.DATA_ROOT, file_name))
                # np_image = np.array(image.resize( np.array(image.size) / Data.IMAGE_SCALE ))
                np_image = np.array(image.resize(np.array(Data.RESIZE_SIZE)))
                # self.__data.append([image, self.__y[y_file_name]])

                self.__total_size += 1

                if img_no not in self.__data_dict:
                    self.__data_dict[img_no] = []
                self.__data_dict[img_no].append([np_image, self.__get_same_size_mask(image, y_file_name)])

        for img_no, data_list in self.__data_dict.items():
            self.__data_list.append([img_no, data_list])

        self.__data_list.sort(key=lambda x: self.__sort_list.index(x[0]) if self.__sort_list else x[0])

        self.echo('\nFinish Loading\n')

    ''' 将 mask 图转为 0 1 像素 '''

    @staticmethod
    def __get_mask(file_name):
        mask = Image.open(os.path.join(Data.DATA_ROOT, file_name)).convert('L')
        return mask

    ''' 获取跟 image 相同 size 的 mask '''

    def __get_same_size_mask(self, image, y_file_name):
        mask = self.__y[y_file_name]
        # mask = np.array( mask.resize( np.array(image.size) / Data.IMAGE_SCALE ) )
        mask = np.array(mask.resize(np.array(Data.RESIZE_SIZE)))

        background = copy.deepcopy(mask)
        background[background != 255] = 0
        background[background == 255] = 1

        mask[mask == 255] = 0
        mask[mask > 0] = 1
        return np.array([background, mask]).transpose([1, 2, 0])

    ''' 获取下个 batch '''

    def next_batch(self, batch_size, loop=True):
        if not loop and self.__cur_index >= self.__data_len:
            return None, None

        start_index = self.__cur_index
        end_index = self.__cur_index + batch_size
        left_num = 0

        if end_index >= self.__data_len:
            left_num = end_index - self.__data_len
            end_index = self.__data_len

        X, y = zip(*self.__data[start_index: end_index])

        if not loop:
            self.__cur_index = end_index
            return np.array(X), np.array(y)

        if not left_num:
            self.__cur_index = end_index if end_index < self.__data_len else 0
            return np.array(X), np.array(y)

        while left_num:
            end_index = left_num
            if end_index > self.__data_len:
                left_num = end_index - self.__data_len
                end_index = self.__data_len
            else:
                left_num = 0

            left_x, left_y = zip(*self.__data[: end_index])
            X += left_x
            y += left_y

        self.__cur_index = end_index if end_index < self.__data_len else 0
        return np.array(X), np.array(y)

    ''' 获取数据集大小 '''

    def get_size(self):
        return self.__data_len

    ''' 重置当前 index 位置 '''

    def reset_cur_index(self):
        self.__cur_index = 0

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
                print (msg)

    @staticmethod
    def get_sort_list():
        img_no_set = set()
        for i, file_name in enumerate(os.listdir(Data.DATA_ROOT)):
            if os.path.splitext(file_name)[1].lower() != '.png':
                continue

            img_no = os.path.splitext(file_name)[0]
            img_no_set.add(img_no)

        img_no_list = list(img_no_set)
        random.shuffle(img_no_list)
        return img_no_list

# Download.run()

# train_data = Data(0.0, 0.64)
# for i in range(4):
#     batch_x , batch_y = train_data.next_batch(1)
#     # #
#     print '********************************'
#     print train_data.get_size()
#     print batch_x.shape
#     print batch_y.shape
