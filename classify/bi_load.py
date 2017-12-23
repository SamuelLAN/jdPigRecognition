#!/usr/bin/Python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import time
import random
import numpy as np
from PIL import Image
import threading

if '2.7' in sys.version:
    import Queue as queue
else:
    import queue


class Data:
    DATA_ROOT = r'../data/TrainImgMore'
    RESIZE = [224, 224]
    # RESIZE = [39, 39]
    RATIO = 1.0
    NUM_CLASSES = 2

    def __init__(self, pig_id, start_ratio=0.0, end_ratio=1.0, name='', resize=None):
        self.__chang_dir()

        # 初始化变量
        self.__pig_id = pig_id
        self.__name = name
        self.__same_data = []
        self.__diff_data = []
        self.__data = {}
        self.__resize = resize if resize else self.RESIZE

        # 加载全部数据
        self.__load()

        # 检查输入参数
        start_ratio = min(max(0.0, start_ratio), 1.0)
        end_ratio = min(max(0.0, end_ratio), 1.0)

        for pig_id, pig_list in self.__data.items():
            pig_len = len(pig_list)

            # 根据比例计算数据的位置范围
            start_index = int(pig_len * start_ratio)
            end_index = int(pig_len * end_ratio)
            new_pig_list = pig_list[start_index: end_index]

            if int(pig_id) == self.__pig_id:
                self.__same_data += new_pig_list
            else:
                self.__diff_data += new_pig_list

        del self.__data

        self.__same_len = len(self.__same_data)
        self.__diff_len = len(self.__diff_data)

        self.__same_len = len(self.__same_data)
        self.__diff_len = len(self.__diff_data)
        self.__data_len = int(2 * self.__same_len)

        random.shuffle(self.__same_data)
        random.shuffle(self.__diff_data)

        self.__queue = queue.Queue()
        self.__stop_thread = False
        self.__thread = None

        self.__cur_index = 0

    @staticmethod
    def __chang_dir():
        # 将运行路径切换到当前文件所在路径
        cur_dir_path = os.path.split(__file__)[0]
        if cur_dir_path and os.path.abspath(os.path.curdir) != os.path.abspath(cur_dir_path):
            os.chdir(cur_dir_path)
            sys.path.append(cur_dir_path)

    ''' 加载数据 '''

    def __load(self):
        self.echo('Loading %s_%d data ...' % (self.__name, self.__pig_id))
        file_list = os.listdir(self.DATA_ROOT)
        file_len = len(file_list)

        for i, file_name in enumerate(file_list):
            progress = float(i + 1) / file_len * 100
            self.echo('\r >> progress: %.2f%% \t' % progress, False)

            split_file_name = os.path.splitext(file_name)
            no_list = split_file_name[0].split('_')

            if split_file_name[1].lower() != '.jpg' or int(no_list[-1]) == 1:
                continue

            pig_id = int(split_file_name[0].split('_')[0]) - 1
            file_path = os.path.join(self.DATA_ROOT, file_name)

            if pig_id not in self.__data:
                self.__data[pig_id] = []
            self.__data[pig_id].append([split_file_name[0], file_path])

        self.echo(' sorting data ... ')
        for pig_id, pig_list in self.__data.items():
            pig_list.sort(key=lambda x: x[0])

        self.echo('\nFinish Loading\n')

    def __get_data(self):
        max_q_size = min(self.__data_len, 500)
        while not self.__stop_thread:
            while self.__queue.qsize() <= max_q_size:
                y = random.randint(0, 1)
                if y == 1:
                    file_name, img_path = self.__same_data[self.__cur_index]
                    self.__cur_index = (self.__cur_index + 1) % self.__same_len
                else:
                    file_name, img_path = self.__diff_data[random.randrange(0, self.__diff_len)]

                x, y = self.__get_x_y(img_path, y)
                self.__queue.put([x, y])

            time.sleep(0.3)

        while self.__queue.qsize() > 0:
            self.__queue.get()

        self.echo(
            '\n*************************************\n Thread "get_%s_data" stop\n***********************\n' % self.__name)

    def start_thread(self):
        self.__stop_thread = False
        self.__thread = threading.Thread(target=self.__get_data, name=('get_%s_data' % self.__name))
        self.__thread.start()
        self.echo('Thread "get_%s_data" is running ... ' % self.__name)

    def stop(self):
        self.__stop_thread = True

    @staticmethod
    def __get_x_y(img_path, y):
        label = np.zeros([Data.NUM_CLASSES])
        label[y] = 1

        return Data.add_padding(img_path), label

    # @staticmethod
    # def __get_three_patch(img_path):
    #     np_image = np.array( Image.open(img_path) )
    #     h, w, c = np_image.shape
    #
    #     if h > w:
    #         _size = w
    #         padding = int( (h - _size) / 2 )
    #         np_image_1 = np_image[:_size, :, :]
    #         np_image_2 = np_image[padding: padding + _size, :, :]
    #         np_image_3 = np_image[-_size:, :, :]
    #
    #     else:
    #         _size = h
    #         padding = int( (w - _size) / 2 )
    #         np_image_1 = np_image[:, :_size, :]
    #         np_image_2 = np_image[:, padding: padding + _size, :]
    #         np_image_3 = np_image[:, -_size:, :]
    #
    #     return [Data.__resize_np_img(np_image_1), Data.__resize_np_img(np_image_2), Data.__resize_np_img(np_image_3)]

    def __resize_np_img(self, np_image):
        return np.array(Image.fromarray(np_image).resize(self.__resize), dtype=np.float32)

    def add_padding(self, img_path):
        image = Image.open(img_path)
        w, h = image.size
        ratio = float(w) / h

        if abs(ratio - Data.RATIO) <= 0.1:
            return np.array(image.resize(self.__resize))

        np_image = np.array(image)
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

        new_image = Image.fromarray(np.cast['uint8'](np_new_image))
        return np.array(new_image.resize(self.__resize))

    def next_batch(self, batch_size):
        X = []
        y = []
        for i in range(batch_size):
            while self.__queue.empty():
                time.sleep(0.1)
            if not self.__queue.empty():
                _x, _y = self.__queue.get()
                X.append(_x)
                y.append(_y)
        return np.array(X), np.array(y)

    ''' 获取数据集大小 '''

    def get_size(self):
        return self.__data_len

    ''' 输出展示 '''

    @staticmethod
    def echo(msg, crlf=True):
        if crlf:
            print(msg)
        else:
            sys.stdout.write(msg)
            sys.stdout.flush()


class TestData:
    DATA_ROOT = r'../data/TrainImgMore'
    RESIZE = [224, 224]
    # RESIZE = [39, 39]
    RATIO = 1.0
    NUM_CLASSES = 30

    def __init__(self, start_ratio=0.0, end_ratio=1.0, name='', resize=None):
        self.__chang_dir()

        # 初始化变量
        self.__name = name
        self.__data = {}
        self.__data_list = []
        self.__resize = resize if resize else self.RESIZE

        # 加载全部数据
        self.__load()

        # 检查输入参数
        start_ratio = min(max(0.0, start_ratio), 1.0)
        end_ratio = min(max(0.0, end_ratio), 1.0)

        for pig_id, pig_list in self.__data.items():
            pig_len = len(pig_list)

            # 根据比例计算数据的位置范围
            start_index = int(pig_len * start_ratio)
            end_index = int(pig_len * end_ratio)
            new_pig_list = pig_list[start_index: end_index]

            self.__data_list += new_pig_list

        del self.__data

        self.__data_len = len(self.__data_list)
        random.shuffle(self.__data_list)

        self.__queue = queue.Queue()

        self.__cur_index = 0

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
        file_list = os.listdir(self.DATA_ROOT)
        file_len = len(file_list)

        for i, file_name in enumerate(file_list):
            progress = float(i + 1) / file_len * 100
            self.echo('\r >> progress: %.2f%% \t' % progress, False)

            split_file_name = os.path.splitext(file_name)
            no_list = split_file_name[0].split('_')

            if split_file_name[1].lower() != '.jpg' or int(no_list[-1]) == 1:
                continue

            pig_id = int(split_file_name[0].split('_')[0]) - 1
            file_path = os.path.join(self.DATA_ROOT, file_name)

            if pig_id not in self.__data:
                self.__data[pig_id] = []
            self.__data[pig_id].append([split_file_name[0], file_path])

        self.echo(' sorting data ... ')
        for pig_id, pig_list in self.__data.items():
            pig_list.sort(key=lambda x: x[0])

        self.echo('\nFinish Loading\n')

    @staticmethod
    def __get_x_y(img_path):
        return Data.add_padding(img_path), TestData.__get_y(img_path)

    @staticmethod
    def __get_y(img_path):
        no_list = os.path.splitext(os.path.split(img_path)[1])[0].split('_')

        pig_no = int(no_list[0]) - 1
        label = np.zeros([TestData.NUM_CLASSES])
        label[pig_no] = 1

        return label

    def __resize_np_img(self, np_image):
        return np.array(Image.fromarray(np_image).resize(self.__resize), dtype=np.float32)

    def add_padding(self, img_path):
        image = Image.open(img_path)
        w, h = image.size
        ratio = float(w) / h

        if abs(ratio - Data.RATIO) <= 0.1:
            return np.array(image.resize(self.__resize))

        np_image = np.array(image)
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

        new_image = Image.fromarray(np.cast['uint8'](np_new_image))
        return np.array(new_image.resize(self.__resize))

    def __read_img_list(self, img_list):
        X = []
        Y = []

        for img_path in img_list:
            x, y = self.__get_x_y(img_path)
            X.append(x)
            Y.append(y)

        return np.array(X), np.array(Y)

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

        _, path_list = zip(*self.__data_list[start_index: end_index])

        if not loop:
            self.__cur_index = end_index
            return self.__read_img_list(path_list)

        if not left_num:
            self.__cur_index = end_index if end_index < self.__data_len else 0
            return self.__read_img_list(path_list)

        while left_num:
            end_index = left_num
            if end_index > self.__data_len:
                left_num = end_index - self.__data_len
                end_index = self.__data_len
            else:
                left_num = 0

            _, left_path_list = zip(*self.__data_list[: end_index])
            path_list += left_path_list

        self.__cur_index = end_index if end_index < self.__data_len else 0
        return self.__read_img_list(path_list)

    def get_label_list(self):
        y = []
        for _, img_path in self.__data_list:
            y.append(self.__get_y(img_path))
        return np.array(y)

    ''' 获取数据集大小 '''

    def get_size(self):
        return self.__data_len

    def reset_cur_index(self):
        self.__cur_index = 0

    ''' 输出展示 '''

    @staticmethod
    def echo(msg, crlf=True):
        if crlf:
            print(msg)
        else:
            sys.stdout.write(msg)
            sys.stdout.flush()


class TestBData:
    DATA_ROOT = r'../data/Test_B'
    RESIZE = [224, 224]
    # RESIZE = [39, 39]
    RATIO = 1.0
    NUM_CLASSES = 30

    def __init__(self, resize=None):
        self.__chang_dir()

        # 初始化变量
        self.__data = []
        self.__resize = resize if resize else self.RESIZE

        # 加载全部数据
        self.__load()

        self.__data_len = len(self.__data)

        self.__queue = queue.Queue()

        self.__cur_index = 0

    @staticmethod
    def __chang_dir():
        # 将运行路径切换到当前文件所在路径
        cur_dir_path = os.path.split(__file__)[0]
        if cur_dir_path and os.path.abspath(os.path.curdir) != os.path.abspath(cur_dir_path):
            os.chdir(cur_dir_path)
            sys.path.append(cur_dir_path)

    ''' 加载数据 '''

    def __load(self):
        self.echo('Loading Test_B data ...')
        file_list = os.listdir(self.DATA_ROOT)
        file_len = len(file_list)

        for i, file_name in enumerate(file_list):
            progress = float(i + 1) / file_len * 100
            self.echo('\r >> progress: %.2f%% \t' % progress, False)

            split_file_name = os.path.splitext(file_name)
            no_list = split_file_name[0].split('_')

            if split_file_name[1].lower() != '.jpg' or 'pig' not in split_file_name[0].lower() \
                    or 'MACOSX' in split_file_name[0]:
                continue

            pig_id = int(split_file_name[0].split('_')[0])
            file_path = os.path.join(self.DATA_ROOT, file_name)

            self.__data.append([pig_id, file_path])

        self.echo('\nFinish Loading\n')

    @staticmethod
    def __get_x_y(img_path):
        return Data.add_padding(img_path), TestBData.__get_y(img_path)

    @staticmethod
    def __get_y(img_path):
        no_list = os.path.splitext(os.path.split(img_path)[1])[0].split('_')
        return int(no_list[0])

    def __resize_np_img(self, np_image):
        return np.array(Image.fromarray(np_image).resize(self.__resize), dtype=np.float32)

    def add_padding(self, img_path):
        image = Image.open(img_path)
        w, h = image.size
        ratio = float(w) / h

        if abs(ratio - Data.RATIO) <= 0.1:
            return np.array(image.resize(self.__resize))

        np_image = np.array(image)
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

        new_image = Image.fromarray(np.cast['uint8'](np_new_image))
        return np.array(new_image.resize(self.__resize))

    def __read_img_list(self, img_list):
        X = []

        for img_path in img_list:
            x, y = self.__get_x_y(img_path)
            X.append(x)

        return np.array(X)

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

        _, path_list = zip(*self.__data[start_index: end_index])

        if not loop:
            self.__cur_index = end_index
            return self.__read_img_list(path_list)

        if not left_num:
            self.__cur_index = end_index if end_index < self.__data_len else 0
            return self.__read_img_list(path_list)

        while left_num:
            end_index = left_num
            if end_index > self.__data_len:
                left_num = end_index - self.__data_len
                end_index = self.__data_len
            else:
                left_num = 0

            _, left_path_list = zip(*self.__data[: end_index])
            path_list += left_path_list

        self.__cur_index = end_index if end_index < self.__data_len else 0
        return self.__read_img_list(path_list)

    def get_label_list(self):
        y = []
        for _, img_path in self.__data:
            y.append(self.__get_y(img_path))
        return np.array(y)

    ''' 获取数据集大小 '''

    def get_size(self):
        return self.__data_len

    def reset_cur_index(self):
        self.__cur_index = 0

    ''' 输出展示 '''

    @staticmethod
    def echo(msg, crlf=True):
        if crlf:
            print(msg)
        else:
            sys.stdout.write(msg)
            sys.stdout.flush()

# Download.run()

# train_data = Data(0.0, 0.64, 'train')
#
# print 'size:'
# print train_data.get_size()
#
# for i in range(10):
#     batch_x, batch_y = train_data.next_batch(10)
#
#     print '\n*************** %d *****************' % i
#     print train_data.get_size()
#     print batch_x.shape
#     print batch_y.shape
#
#     tmp_x = batch_x[0]
#     o_tmp = Image.fromarray(tmp_x)
#     o_tmp.show()
#
#     time.sleep(1)
#
# train_data.stop()

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
