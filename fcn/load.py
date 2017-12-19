#!/usr/bin/Python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import copy
import random
import zipfile
import numpy as np
from PIL import Image
from six.moves.urllib.request import urlretrieve


'''
    下载数据
'''
class Download:
    URL = 'http://www.lin-baobao.com/pig/data.zip'
    DATA_ROOT = r'data'
    FILE_NAME = 'data.zip'
    EXPECTED_BYTES = 228090460
    FILE_NUM = 1815

    def __init__(self):
        pass


    ''' 将运行路径切换到当前文件所在路径 '''
    @staticmethod
    def __changDir():
        cur_dir_path = os.path.split(__file__)[0]
        if cur_dir_path:
            os.chdir(cur_dir_path)
            sys.path.append(cur_dir_path)


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
            if os.path.splitext(file_name)[1].lower() != '.jpg':
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


'''
 Data: 取数据到基类
 对外提供接口:
    get_size()
    next_batch()
'''
class Data:
    DATA_ROOT = r'data'
    IMAGE_SCALE = 2
    RESIZE_SIZE = [640, 360]

    def __init__(self, start_ratio = 0.0, end_ratio = 1.0, name = '', sort_list = []):
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

            if os.path.splitext(file_name)[1].lower() != '.jpg':
                continue

            if 'mask' in file_name and file_name not in self.__y:
                self.__y[file_name] = self.__get_mask(file_name)

            if 'mask' not in file_name:
                img_no = file_name.split('_')[0]
                y_file_name = img_no + '_mask.jpg'

                if y_file_name not in self.__y:
                    self.__y[y_file_name] = self.__get_mask(y_file_name)

                image = Image.open(os.path.join(self.DATA_ROOT, file_name))
                # np_image = np.array(image.resize( np.array(image.size) / Data.IMAGE_SCALE ))
                np_image = np.array(image.resize( np.array(Data.RESIZE_SIZE) ))
                # self.__data.append([image, self.__y[y_file_name]])

                self.__total_size += 1

                if img_no not in self.__data_dict:
                    self.__data_dict[img_no] = []
                self.__data_dict[img_no].append([np_image, self.__get_same_size_mask(image, y_file_name)])

        iter_items = self.__data_dict.iteritems() if '2.7' in sys.version else self.__data_dict.items()
        for img_no, data_list in iter_items:
            self.__data_list.append([int(img_no), data_list])

        if '2.7' in sys.version:
            self.__data_list.sort(self.__sort) # 按顺序排列
        else:
            self.__data_list.sort(key=lambda x: self.__sort_list.index(x[0]) if self.__sort_list else x[0])

        self.echo('\nFinish Loading\n')


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


    ''' 将 mask 图转为 0 1 像素 '''
    @staticmethod
    def __get_mask(file_name):
        mask = Image.open(os.path.join(Data.DATA_ROOT, file_name)).convert('L')
        return mask


    ''' 获取跟 image 相同 size 的 mask '''
    def __get_same_size_mask(self, image, y_file_name):
        mask = self.__y[y_file_name]
        # mask = np.array( mask.resize( np.array(image.size) / Data.IMAGE_SCALE ) )
        mask = np.array( mask.resize( np.array(Data.RESIZE_SIZE) ) )

        background = copy.deepcopy(mask)
        background[background != 255] = 0
        background[background == 255] = 1

        mask[mask == 255] = 0
        mask[mask > 0] = 1
        return np.array([background, mask]).transpose([1, 2, 0])


    ''' 获取下个 batch '''
    def next_batch(self, batch_size, loop = True):
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
            print (msg)
        else:
            sys.stdout.write(msg)
            sys.stdout.flush()


    @staticmethod
    def get_sort_list():
        img_no_set = set()
        for i, file_name in enumerate(os.listdir(Data.DATA_ROOT)):
            if os.path.splitext(file_name)[1].lower() != '.jpg':
                continue

            img_no = int(file_name.split('_')[0])
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
