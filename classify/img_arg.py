#!/usr/bin/Python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import copy
import random
import numpy as np
from PIL import Image
from PIL import ImageEnhance

# 将运行路径切换到当前文件所在路径
cur_dir_path = os.path.abspath(os.path.split(__file__)[0])
if cur_dir_path:
    os.chdir(cur_dir_path)
    sys.path.append(cur_dir_path)
    sys.path.append(os.path.split(cur_dir_path)[0])


class Img:
    IMG_PATH = r'../data/TrainImg'
    IMG_MORE_PATH = r'../data/TrainImgMore'

    NUM_TRANSFORM = 12
    NUM_BLOCK_IMAGE = 4
    NUM_CORP_IMAGE = 4
    MIN_BLOCK_PIG_RATIO = 0.35

    RESIZE_SIZE = [640, 360]

    def __init__(self):
        self.__img_list = []
        self.__alreadyList = {}
        self.__progress_index = 0
        self.__progress_len = 0

    def __check_folder(self):
        data_dir = os.path.split(self.IMG_PATH)[0]
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)

        if not os.path.isdir(self.IMG_PATH):
            os.mkdir(self.IMG_PATH)

        if not os.path.isdir(self.IMG_MORE_PATH):
            os.mkdir(self.IMG_MORE_PATH)

    ''' 检查文件夹已经存在的 patch ，避免重复生成 '''

    def __get_already_exist_list(self):
        already_list = {}
        for file_name in os.listdir(self.IMG_MORE_PATH):
            split_file_name = os.path.splitext(file_name)
            if split_file_name[1].lower() != '.jpg':
                continue
            file_name = split_file_name[0]
            file_no = file_name.split('_')

            img_name = '%s_%s.jpg' % (file_no[0], file_no[1])
            if img_name not in already_list:
                already_list[img_name] = 0
            already_list[img_name] += 1

            if already_list[img_name] >= (self.NUM_TRANSFORM + self.NUM_BLOCK_IMAGE + self.NUM_CORP_IMAGE):
                self.__alreadyList[img_name] = True

    ''' 获取图片列表 '''

    def __get_img_list(self):
        for file_name in os.listdir(self.IMG_PATH):
            split_file_name = os.path.splitext(file_name)
            if split_file_name[1].lower() != '.jpg' or 'pig' not in split_file_name[0].lower() \
                    or file_name in self.__alreadyList:
                continue

            self.__img_list.append(os.path.join(self.IMG_PATH, file_name))

        # self.__progress_len = len(self.__img_list) * (self.NUM_TRANSFORM + self.NUM_BLOCK_IMAGE + self.NUM_CORP_IMAGE)
        self.__progress_len = len(self.__img_list)

    ''' 用最小的框 框住猪并生成猪的图片 '''

    @staticmethod
    def __get_pig_object(image):
        np_image = np.array(image.convert('L'))
        w, h = np_image.shape

        min_w = -1
        max_w = w
        min_h = -1
        max_h = h

        for i in range(w):
            for j in range(h):
                if np_image[i][j] != 0:
                    min_w = i
                    break
            if min_w >= 0:
                break

        for j in range(h):
            for i in range(w):
                if np_image[i][j] != 0:
                    min_h = j
                    break
            if min_h >= 0:
                break

        for i in range(w):
            for j in range(h):
                if np_image[w - i - 1][j] != 0:
                    max_w = w - i - 1
                    break
            if max_w < w:
                break

        for j in range(h):
            for i in range(w):
                if np_image[i][h - j - 1] != 0:
                    max_h = h - j - 1
                    break
            if max_h < h:
                break

        min_w = max(min_w, 0)
        min_h = max(min_h, 0)
        max_w = min(max_w, w - 1)
        max_h = min(max_h, h - 1)

        np_image = np.array(image)
        np_pig = np_image[min_w: max_w + 1, min_h: max_h + 1, :]

        return Image.fromarray(np_pig), [min_w, max_w, min_h, max_h]

    ''' 获取更多的图片 '''

    def __get_more_img(self, img_path):
        im_name = os.path.splitext(os.path.split(img_path)[1])[0]
        im_name = im_name.replace('_pig', '')
        file_no = 0

        self.__cal_progress(im_name)  # 输出进度

        origin_image = Image.open(img_path)

        # 生成猪的原图
        image, pos = self.__get_pig_object(origin_image)
        image.save(os.path.join(self.IMG_MORE_PATH, '%s_%d.jpg' % (im_name, file_no)))

        if pos[1] - pos[0] < 50 or pos[3] - pos[2] < 50:
            return

        # 生成能用最小的框框住猪的原图(带背景)
        file_no += 1
        frame_img = Image.open(os.path.join(os.path.split(img_path)[0], '%s.jpg' % im_name))
        np_frame_img = np.array(frame_img.resize(self.RESIZE_SIZE))
        np_frame_img = np_frame_img[pos[0]: pos[1] + 1, pos[2]: pos[3] + 1]
        new_frame_img = Image.fromarray(np_frame_img)
        new_frame_img.save(os.path.join(self.IMG_MORE_PATH, '%s_%d.jpg' % (im_name, file_no)))

        # 水平翻转
        file_no += 1
        flip_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        flip_image.save(os.path.join(self.IMG_MORE_PATH, '%s_%d.jpg' % (im_name, file_no)))

        # 垂直翻转
        file_no += 1
        flip_image = image.transpose(Image.FLIP_TOP_BOTTOM)
        flip_image.save(os.path.join(self.IMG_MORE_PATH, '%s_%d.jpg' % (im_name, file_no)))

        # 亮度
        brightness_up = 1 + random.random() * 0.8
        brightness_down = 1 - random.random() * 0.85
        enh_bri = ImageEnhance.Brightness(image)

        # 亮度增强
        file_no += 1
        image_brightened = enh_bri.enhance(brightness_up)
        image_brightened.save(os.path.join(self.IMG_MORE_PATH, '%s_%d.jpg' % (im_name, file_no)))

        # 亮度降低
        file_no += 1
        image_brightened = enh_bri.enhance(brightness_down)
        image_brightened.save(os.path.join(self.IMG_MORE_PATH, '%s_%d.jpg' % (im_name, file_no)))

        # 色度
        color_up = 1 + random.random() * 0.8
        color_down = 1 - random.random() * 0.7
        enh_col = ImageEnhance.Color(image)

        # 色度增强
        file_no += 1
        image_colored = enh_col.enhance(color_up)
        image_colored.save(os.path.join(self.IMG_MORE_PATH, '%s_%d.jpg' % (im_name, file_no)))

        # 色度降低
        file_no += 1
        image_colored = enh_col.enhance(color_down)
        image_colored.save(os.path.join(self.IMG_MORE_PATH, '%s_%d.jpg' % (im_name, file_no)))

        # 对比度
        contrast_up = 1 + random.random() * 0.55
        contrast_down = 1 - random.random() * 0.5
        enh_con = ImageEnhance.Contrast(image)

        # 对比度增强
        file_no += 1
        image_contrasted = enh_con.enhance(contrast_up)
        image_contrasted.save(os.path.join(self.IMG_MORE_PATH, '%s_%d.jpg' % (im_name, file_no)))

        # 对比度降低
        file_no += 1
        image_contrasted = enh_con.enhance(contrast_down)
        image_contrasted.save(os.path.join(self.IMG_MORE_PATH, '%s_%d.jpg' % (im_name, file_no)))

        # 锐度
        sharpness_up = 1 + random.random() * 2
        sharpness_down = 1 - random.random() * 0.8
        enh_sha = ImageEnhance.Sharpness(image)

        # 锐度增强
        file_no += 1
        image_sharped = enh_sha.enhance(sharpness_up)
        image_sharped.save(os.path.join(self.IMG_MORE_PATH, '%s_%d.jpg' % (im_name, file_no)))

        # 锐度降低
        file_no += 1
        image_sharped = enh_sha.enhance(sharpness_down)
        image_sharped.save(os.path.join(self.IMG_MORE_PATH, '%s_%d.jpg' % (im_name, file_no)))

        # 遮挡
        np_image = np.array(image)
        for i in range(self.NUM_BLOCK_IMAGE):
            block_image = Image.fromarray(self.__get_block_img(copy.deepcopy(np_image)))

            file_no += 1
            block_image.save(os.path.join(self.IMG_MORE_PATH, '%s_%d.jpg' % (im_name, file_no)))

        # 随机裁剪图片
        for i in range(self.NUM_CORP_IMAGE):
            corp_image = self.__random_corp(np_image)
            file_no += 1
            corp_image.save(os.path.join(self.IMG_MORE_PATH, '%s_%d.jpg' % (im_name, file_no)))

    @staticmethod
    def __random_corp(np_image):
        w, h, c = np_image.shape

        corp_w = int(float(random.randrange(3, 9)) / 10 * w)
        corp_h = int(float(random.randrange(3, 9)) / 10 * h)

        x1 = random.randrange(0, int(w - corp_w))
        y1 = random.randrange(0, int(h - corp_h))

        corp_image = np_image[x1: x1 + corp_w, y1: y1 + corp_h, :]
        return Image.fromarray(corp_image)

    @staticmethod
    def __get_block_img(block_image):
        w, h, c = block_image.shape

        def __get_img():
            ratio = 1.0 / random.randint(8, 12)
            block_w = int(w * ratio)
            block_h = int(h * ratio)

            _half_block_w = int(block_w / 2)
            _half_block_h = int(block_h / 2)

            _block_center_x = random.randrange(_half_block_w, w - _half_block_w)
            _block_center_y = random.randrange(_half_block_h, h - _half_block_h)

            _block_content = block_image[_block_center_x - _half_block_w: _block_center_x + _half_block_w,
                             _block_center_y - _half_block_h: _block_center_y + _half_block_h, :]

            return _block_content, _block_center_x, _block_center_y, _half_block_w, _half_block_h

        block_content, block_center_x, block_center_y, half_block_w, half_block_h = __get_img()

        block_image[block_center_x - half_block_w: block_center_x + half_block_w,
        block_center_y - half_block_h: block_center_y + half_block_h, :] = np.array([0, 0, 0], np.int8)
        return block_image

    ''' 获取进度 '''

    def __cal_progress(self, img_name, increment=1):
        self.__progress_index += increment
        progress = float(self.__progress_index) / self.__progress_len * 100
        self.echo('\r >> progress: %.2f%% \t processing: %s \t ' % (progress, img_name), False)

    ''' 输出展示 '''

    @staticmethod
    def echo(msg, crlf=True):
        if crlf:
            print(msg)
        else:
            sys.stdout.write(msg)
            sys.stdout.flush()

    def run(self):
        self.__check_folder()

        # self.echo('\nGetting already exist list ...')
        # self.__get_already_exist_list()
        # self.echo('Finish getting already exist list ')

        self.echo('\nGetting img list ...')
        self.__get_img_list()
        self.echo('Finish getting img list ')

        self.echo('\nGetting more img ...')
        for i, img_path in enumerate(self.__img_list):
            self.__get_more_img(img_path)
        self.echo('Finish getting more img ')

        self.echo('done')


o_img = Img()
o_img.run()
