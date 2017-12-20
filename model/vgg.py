#!/usr/bin/Python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import numpy as np
from six.moves.urllib.request import urlretrieve

''' VGG 模型 (16层版) '''


class VGG:
    MODEL_16 = r'vgg16.npy'
    MODEL_19 = r'vgg19.npy'
    MODEL_16_URL = r'http://www.lin-baobao.com/model/vgg16.npy'
    MODEL_19_URL = r'http://www.lin-baobao.com/model/vgg19.npy'

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
    def load(model_19=False):
        '''
        Returns:
            vgg_mode (dict)
        '''

        cur_dir = VGG.__change_dir()  # 切回当前目录到文件所在目录

        model = VGG.MODEL_16 if not model_19 else VGG.MODEL_19
        model_url = VGG.MODEL_16_URL if not model_19 else VGG.MODEL_19_URL

        if not os.path.isfile(model):
            print('Start downloading %s' % model)
            file_path, _ = urlretrieve(model_url, model, reporthook=VGG.__download_progress)
            stat_info = os.stat(file_path)
            print('\nSuccesfully downloaded %s %d bytes' % (model_url, stat_info.st_size))
        model = np.load(model, encoding='latin1').item()

        os.chdir(cur_dir)  # 切回原来的目录

        return model

    ''' 下载的进度 '''

    @staticmethod
    def __download_progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %.1f%%' % (float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
