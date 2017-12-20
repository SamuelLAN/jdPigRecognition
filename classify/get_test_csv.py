#!/usr/bin/Python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import csv
import numpy as np

# 将运行路径切换到当前文件所在路径
cur_dir_path = os.path.abspath(os.path.split(__file__)[0])
if cur_dir_path:
    os.chdir(cur_dir_path)
    sys.path.append(cur_dir_path)
    sys.path.append(os.path.split(cur_dir_path)[0])

import load
import vgg16_net as vgg


class GetCSV:
    IMG_DIR = r'../data/Test_B'
    RESULT_DIR = r'../result'
    RESULT_FILE_PATH = r'../result/test_B.csv'

    def __init__(self):
        self.__img_list = []
        self.__img_len = 0
        self.__data = {}

        self.__o_vgg = vgg.VGG16(True)

    def __get_image_list(self):
        self.echo('\nGetting image list ...')

        file_list = os.listdir(self.IMG_DIR)
        file_list_len = len(file_list)

        for i, file_name in enumerate(file_list):
            progress = float(i + 1) / file_list_len * 100
            self.echo('\r  >> progress; %.6f ' % progress, False)

            split_file_name = os.path.splitext(file_name)
            if split_file_name[1].lower() != '.jpg' or 'pig' not in split_file_name[0].lower() \
                    or 'MACOSX' in split_file_name[0]:
                continue

            self.__img_list.append([split_file_name[0], os.path.join(self.IMG_DIR, file_name)])

        self.__img_len = len(self.__img_list)

        self.echo('Finish getting image list')

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

    def __predict(self):
        self.echo('\nStart predicting pig ... ')

        for i, (file_name, img_path) in enumerate(self.__img_list):
            progress = float(i + 1) / self.__img_len * 100
            self.echo('\r Progress: %.2f | %d / %d \t ' % (progress, i + 1, self.__img_len), False)

            pig_no = int(file_name.split('_')[0])
            np_image = load.Data.add_padding(img_path)
            output = self.__o_vgg.use_model(np_image)

            prob = self.softmax(output)
            self.__data[pig_no] = prob

        self.echo('Finish predicting ')

    def __save_result(self):
        if not os.path.isdir(self.RESULT_DIR):
            os.mkdir(self.RESULT_DIR)

        self.echo('\nSaving result to %s ... ' % self.RESULT_FILE_PATH)
        data_len = len(self.__data)

        with open(self.RESULT_FILE_PATH, 'w') as f:
            writer = csv.writer(f)

            count = 0
            for pig_no, predict_prob in self.__data.items():
                count += 1
                progress = float(count) / data_len * 100.0
                self.echo('\r  >> progress: %.6f ' % progress, False)

                for i, prob in enumerate(predict_prob):
                    writer.writerow([pig_no, i + 1, '%.10f' % prob])

        self.echo('Finish saving result ')

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    def run(self):
        self.__get_image_list()

        self.__predict()

        self.__save_result()

        self.echo('\ndone')


o_get_csv = GetCSV()
o_get_csv.run()
