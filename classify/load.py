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
if '2.7' in sys.version:
    import Queue as queue
else:
    import queue
import time


'''
    下载数据
'''
class Download:
    URL = 'http://www.lin-baobao.com/pig/TrainImg.zip'
    DATA_ROOT = r'data/TrainImg'
    FILE_NAME = 'TrainImg.zip'
    EXPECTED_BYTES = 973849767
    FILE_NUM = 2986

    def __init__(self):
        pass


    ''' 将运行路径切换到当前文件所在路径 '''
    @staticmethod
    def __changDir():
        cur_dir_path = os.path.split(__file__)[0]
        if cur_dir_path and os.path.abspath( os.path.curdir ) != os.path.abspath(cur_dir_path):
            os.chdir(cur_dir_path)
            sys.path.append(cur_dir_path)

        # mkdir ./data
        if not os.path.isdir( os.path.split(Download.DATA_ROOT)[0] ):
            os.mkdir(os.path.split(Download.DATA_ROOT)[0])


    ''' 下载进度 '''
    @staticmethod
    def __downloadProgressHook(count, block_size, total_size):
        sys.stdout.write('\r >> Downloading %.1f%%' % (float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()


    ''' 判断是否需要下载；若需，下载数据压缩包 '''
    @staticmethod
    def __maybeDownload(force=False):
        """Download a file if not present, and make sure it's the right size."""
        if not os.path.isdir(Download.DATA_ROOT):
            os.mkdir(Download.DATA_ROOT)
        file_path = os.path.join(Download.DATA_ROOT, Download.FILE_NAME)
        if force or not os.path.exists(file_path):
            print ('Attempting to download: %s' % Download.FILE_NAME)
            filename, _ = urlretrieve(Download.URL, file_path, reporthook=Download.__downloadProgressHook)
            print ('\nDownload Complete!')
        stat_info = os.stat(file_path)
        if stat_info.st_size == Download.EXPECTED_BYTES:
            print ('Found and verified %s' % file_path)
        else:
            raise Exception(
                'Failed to verify ' + file_path + '. Can you get to it with a browser?')


    @staticmethod
    def __checkFileNum():
        if not os.path.isdir(Download.DATA_ROOT):
            return False
        file_num = 0
        for file_name in os.listdir(Download.DATA_ROOT):
            split_file_name = os.path.splitext(file_name)
            if split_file_name[1].lower() != '.jpg' or 'pig' in split_file_name[0].lower():
                continue
            file_num += 1
        if file_num != Download.FILE_NUM:
            return False
        return True


    @staticmethod
    def __maybeExtract(force=False):
        file_path = os.path.join(Download.DATA_ROOT, Download.FILE_NAME)

        zip_files = zipfile.ZipFile(file_path, 'r')
        for filename in zip_files.namelist():
            if '__MACOSX' in filename:
                continue
            print ('\t extracting %s ...' % filename)
            data = zip_files.read(filename)
            with open(os.path.join(Download.DATA_ROOT, filename), 'wb') as f:
                f.write(data)


    @staticmethod
    def run():
        Download.__changDir()   # 将路径切换到当前路径

        if Download.__checkFileNum():
            print ('data exist in %s' % Download.DATA_ROOT)
            return

        Download.__maybeDownload()

        print ('Extracting data ...')

        Download.__maybeExtract()

        print ('Finish Extracting')

        print ('done')



class Data:
    DATA_ROOT = r'data/TrainImgMore'
    RESIZE = [224, 224]
    # RESIZE = [39, 39]
    RATIO = 1.0
    NUM_CLASSES = 30

    def __init__(self, start_ratio = 0.0, end_ratio = 1.0, name = '', sort_list = []):
        self.__chang_dir()

        # 初始化变量
        self.__name = name
        self.__data = []
        self.__sort_list = sort_list

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

        # scale = 3
        # data_list = []
        # for i, data in enumerate(self.__data):
        #     if i % scale == 0:
        #         data_list.append(data)
        # self.__data = data_list

        self.__data_len = len(self.__data)
        random.shuffle(self.__data)

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

            self.__data.append( [split_file_name[0], os.path.join(self.DATA_ROOT, file_name)] )
            #
            # pig_bg_file_path = os.path.join(self.DATA_ROOT, '%s_%s_1.jpg' % (no_list[0], no_list[1]))
            # pig_file_path = os.path.join(self.DATA_ROOT, file_name)
            #
            # if not os.path.isfile(pig_file_path):
            #     continue
            #
            # # np_pig = self.add_padding(pig_file_path)
            # # np_pig_bg = self.add_padding(pig_bg_file_path)
            #
            # pig_patch_list = self.__get_three_patch(pig_file_path)
            # pig_bg_patch_list = self.__get_three_patch(pig_bg_file_path)
            # patch_list = pig_patch_list + pig_bg_patch_list
            #
            # pig_no = int(no_list[0]) - 1
            # label = np.zeros([Data.NUM_CLASSES])
            # label[pig_no] = 1
            #
            # self.__data.append([split_file_name[0], patch_list, label])

        # self.echo(' sorting data ... ')
        # self.__data.sort(self.__sort)

        self.echo('\nFinish Loading\n')


    def __get_data(self):
        max_q_size = min(self.__data_len, 500)
        while not self.__stop_thread:
            while self.__queue.qsize() <= max_q_size:
                file_name, img_path = self.__data[self.__cur_index]
                x, y = self.__get_x_y(img_path)

                self.__queue.put([x, y])
                self.__cur_index = (self.__cur_index + 1) % self.__data_len

            time.sleep(0.3)

        self.echo('\n*************************************\n Thread "get_%s_data" stop\n***********************\n' % self.__name)


    def start_thread(self):
        self.__thread = threading.Thread(target=self.__get_data, name=('get_%s_data' % self.__name))
        self.__thread.start()
        self.echo('Thread "get_%s_data" is running ... ' % self.__name)


    def stop(self):
        self.__stop_thread = True


    @staticmethod
    def __get_x_y(img_path):
        no_list = os.path.splitext(os.path.split(img_path)[1])[0].split('_')

        pig_no = int(no_list[0]) - 1
        label = np.zeros([Data.NUM_CLASSES])
        label[pig_no] = 1

        return Data.add_padding(img_path), label


    # @staticmethod
    # def __read_img_list(img_list):
    #     X = []
    #     y = []
    #
    #     for img_path in img_list:
    #         no_list = os.path.splitext( os.path.split(img_path)[1] )[0].split('_')
    #
    #         pig_no = int(no_list[0]) - 1
    #         label = np.zeros([Data.NUM_CLASSES])
    #         label[pig_no] = 1
    #
    #         X.append( Data.add_padding(img_path) )
    #         y.append( label )
    #
    #     return np.array(X), np.array(y)
    #
    #
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


    @staticmethod
    def __resize_np_img(np_image):
        return np.array( Image.fromarray(np_image).resize( Data.RESIZE ), dtype=np.float32 )

    
    @staticmethod
    def add_padding(img_path):
        image = Image.open(img_path)
        w, h = image.size
        ratio = float(w) / h

        if abs(ratio - Data.RATIO) <= 0.1:
            return np.array( image.resize( Data.RESIZE ) )

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

        new_image = Image.fromarray( np.cast['uint8'](np_new_image) )
        return np.array( new_image.resize( Data.RESIZE ) )


    def __sort(self, a, b):
        if self.__sort_list:
            index_a = self.__sort_list.index(a[0])
            index_b = self.__sort_list.index(b[0])
            if index_a < index_b:
                return -1
            elif index_a > index_b:
                return 1
            else:
                return 0

        if a[0] < b[0]:
            return -1
        elif a[0] > b[0]:
            return 1
        else:
            return 0


    @staticmethod
    def get_sort_list():
        img_no_set = set()
        for i, file_name in enumerate(os.listdir(Data.DATA_ROOT)):
            split_file_name = os.path.splitext(file_name)
            if split_file_name[1].lower() != '.jpg' or 'pig' in split_file_name[0]:
                continue

            img_no_set.add(split_file_name[0])

        img_no_list = list(img_no_set)
        random.shuffle(img_no_list)
        return img_no_list


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


    # ''' 获取下个 batch '''
    # def next_batch(self, batch_size, loop = True):
    #     if not loop and self.__cur_index >= self.__data_len:
    #         return None, None
    #
    #     start_index = self.__cur_index
    #     end_index = self.__cur_index + batch_size
    #     left_num = 0
    #
    #     if end_index >= self.__data_len:
    #         left_num = end_index - self.__data_len
    #         end_index = self.__data_len
    #
    #     _, path_list = zip(*self.__data[start_index: end_index])
    #
    #     if not loop:
    #         self.__cur_index = end_index
    #         return Data.__read_img_list(path_list)
    #
    #     if not left_num:
    #         self.__cur_index = end_index if end_index < self.__data_len else 0
    #         return Data.__read_img_list(path_list)
    #
    #     while left_num:
    #         end_index = left_num
    #         if end_index > self.__data_len:
    #             left_num = end_index - self.__data_len
    #             end_index = self.__data_len
    #         else:
    #             left_num = 0
    #
    #         _, left_path_list = zip(*self.__data[: end_index])
    #         path_list += left_path_list
    #
    #     self.__cur_index = end_index if end_index < self.__data_len else 0
    #     return Data.__read_img_list(path_list)


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
            print (msg)
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
