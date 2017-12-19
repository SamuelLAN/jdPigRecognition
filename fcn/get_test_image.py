#!/usr/bin/Python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys

# 将运行路径切换到当前文件所在路径
cur_dir_path = os.path.split(__file__)[0]
if cur_dir_path:
    os.chdir(cur_dir_path)
    sys.path.append(cur_dir_path)

from PIL import Image
import numpy as np
import fcn


class GetImage:
    IMG_DIR = r'../deep_id/data/Test_B'
    # RESIZE_SIZE = [640, 360]
    SCALE = 2.0

    def __init__(self):
        self.__img_list = []
        self.__img_len = 0
        self.__o_fcn = fcn.FCN()


    def __get_image_list(self):
        for file_name in os.listdir(self.IMG_DIR):
            split_file_name = os.path.splitext(file_name)
            if split_file_name[1].lower() != '.jpg' or 'pig' in split_file_name[0].lower() \
                    or 'MACOSX' in split_file_name[0]:
                continue

            self.__img_list.append([ split_file_name[0], os.path.join(self.IMG_DIR, file_name) ])

        self.__img_len = len(self.__img_list)


    def __get_pig(self):
        for i, (file_name, img_path) in enumerate(self.__img_list):
            progress = float(i + 1) / self.__img_len * 100
            self.echo('\r Progress: %.2f | %d / %d \t ' % (progress, i + 1, self.__img_len), False)

            image = Image.open(img_path)
            image = np.array(image.resize( np.cast['int32']( np.array(image.size) / self.SCALE ) ))

            np_pig = self.__o_fcn.use_model(image)
            im_pig = Image.fromarray(np_pig)
            im_pig.save( os.path.join(self.IMG_DIR, '%s_pig.jpg' % file_name) )


    ''' 输出展示 '''
    @staticmethod
    def echo(msg, crlf=True):
        if crlf:
            print (msg)
        else:
            sys.stdout.write(msg)
            sys.stdout.flush()


    def run(self):
        self.echo('\nGetting image list ...')
        self.__get_image_list()
        self.echo('Finish getting image list')

        self.echo('\nGetting pig ...')
        self.__get_pig()
        self.echo('Finish getting pig')

        self.echo('\ndone')

o_get_img = GetImage()
o_get_img.run()

