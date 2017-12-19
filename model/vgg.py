#!/usr/bin/Python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import numpy as np
from six.moves.urllib.request import urlretrieve

''' VGG 模型 (16层版) '''


class VGG:
    MODEL = r'vgg16.npy'
    MODEL_URL = r'http://www.lin-baobao.com/model/vgg16.npy'

    def __init__(self):
        pass

    @staticmethod
    def __change_dir():
        cur_dir = os.path.abspath(os.path.curdir)
        chang_dir = os.path.abspath(os.path.split(__file__)[0])
        os.chdir(chang_dir)
        return cur_dir

    ''' 加载模型 '''

    @staticmethod
    def load():
        '''
        Returns:
            vgg_mode (dict)
        '''

        cur_dir = VGG.__change_dir()  # 切回当前目录到文件所在目录

        if not os.path.isfile(VGG.MODEL):
            print('Start downloading %s' % VGG.MODEL)
            file_path, _ = urlretrieve(VGG.MODEL_URL, VGG.MODEL, reporthook=VGG.__download_progress)
            stat_info = os.stat(file_path)
            print('\nSuccesfully downloaded %s %d bytes' % (VGG.MODEL_URL, stat_info.st_size))
        model = np.load(VGG.MODEL, encoding='latin1').item()

        os.chdir(cur_dir)  # 切回原来的目录

        return model

    ''' 下载的进度 '''

    @staticmethod
    def __download_progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (os.path.split(VGG.MODEL)[1],
                                                         float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
