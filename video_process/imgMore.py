#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import sys

# 将运行路径切换到当前文件所在路径
cur_dir_path = os.path.split(__file__)[0]
if cur_dir_path:
    os.chdir(cur_dir_path)
    sys.path.append(cur_dir_path)

import cv2
import random
import numpy as np

import copy
from PIL import Image
from PIL import ImageEnhance


class Img:
    IMG_PATH = r'../data/TrainImg'
    IMG_MORE_PATH = r'../fcn/data/TrainImgMore'

    NUM_TRANSFORM = 12
    NUM_BLOCK_IMAGE = 4
    NUM_CORP_IMAGE = 4
    MIN_BLOCK_PIG_RATIO = 0.35

    def __init__(self):
        self.__imgList = []
        self.__alreadyList = {}
        self.__progressIndex = 0
        self.__progressLen = 0


    ''' 检查文件夹已经存在的 patch ，避免重复生成 '''
    def __getAlreadyExistList(self):
        already_list = {}
        for file_name in os.listdir(self.IMG_MORE_PATH):
            if os.path.splitext(file_name)[1].lower() != '.jpg':
                continue
            file_name = os.path.splitext(file_name)[0]
            file_no = file_name.split('_')

            img_name = '%s_%s.jpg' % (file_no[0], file_no[1])
            if img_name not in already_list:
                already_list[img_name] = 0
            already_list[img_name] += 1

            if already_list[img_name] >= (self.NUM_TRANSFORM + self.NUM_BLOCK_IMAGE + self.NUM_CORP_IMAGE):
                self.__alreadyList[img_name] = True


    ''' 获取图片列表 '''
    def __getImgList(self):
        for file_name in os.listdir(self.IMG_PATH):
            if os.path.splitext(file_name)[1].lower() != '.jpg' or file_name in self.__alreadyList:
                continue

            self.__imgList.append(os.path.join(self.IMG_PATH, file_name))

        self.__progressLen = len(self.__imgList) * (self.NUM_TRANSFORM + self.NUM_BLOCK_IMAGE + self.NUM_CORP_IMAGE)


    def __getSmallPig(self, image):
        np_img = np.array(image)
        w, h, c = np_img.shape

        left_cut = int(0.5 * w)
        cut_img = np_img[:left_cut, :, :]
        cut_border_img = np_img[left_cut - 10: left_cut, :, :]
        while left_cut > 10 and (self.__calPigRatio(cut_img) > 0.1 or self.__calPigRatio(cut_border_img) > 0.01):
            left_cut = int(left_cut * 0.5)
            cut_img = np_img[:left_cut, :, :]
            cut_border_img = np_img[left_cut - 10: left_cut, :, :]

        right_cut = int(0.5 * w) - 1
        cut_img = np_img[right_cut:, :, :]
        cut_border_img = np_img[right_cut: right_cut + 10, :, :]
        while right_cut < w - 10 and (self.__calPigRatio(cut_img) > 0.1 or self.__calPigRatio(cut_border_img) > 0.01):
            right_cut = int( (right_cut + w - 1) * 0.5 )
            cut_img = np_img[right_cut:, :, :]
            cut_border_img = np_img[right_cut: right_cut + 10, :, :]

        top_cut = int(0.5 * h)
        cut_img = np_img[:, : top_cut, :]
        cut_border_img = np_img[: , top_cut - 10: top_cut, :]
        while top_cut > 10 and (self.__calPigRatio(cut_img) > 0.1 or self.__calPigRatio(cut_border_img) > 0.01):
            top_cut = int(top_cut * 0.5)
            cut_img = np_img[:, :top_cut, :]
            cut_border_img = np_img[:, top_cut - 10: top_cut, :]

        bottom_cut = int(0.5 * h) - 1
        cut_img = np_img[:, bottom_cut:, :]
        cut_border_img = np_img[:, bottom_cut: bottom_cut + 10, :]
        while bottom_cut < h - 10 and (self.__calPigRatio(cut_img) > 0.1 or self.__calPigRatio(cut_border_img) > 0.01):
            bottom_cut = int( (bottom_cut + h - 1) * 0.5)
            cut_img = np_img[:, bottom_cut:, :]
            cut_border_img = np_img[:, bottom_cut: bottom_cut + 10, :]

        small_img = np_img[left_cut: right_cut, top_cut: bottom_cut, :]
        return Image.fromarray( small_img )


    def __randomCorp(self, np_image):
        w, h, c = np_image.shape

        def __corpImg():
            corp_w = int( float(random.randrange(3, 9)) / 10 * w)
            corp_h = int( float(random.randrange(3, 9)) / 10 * h)

            x1 = random.randrange(0, int(w - corp_w))
            y1 = random.randrange(0, int(h - corp_h))

            return np_image[x1: x1 + corp_w, y1: y1 + corp_h, :]

        try_times = 0
        corp_image = __corpImg()
        while self.__calPigRatio(corp_image) < 0.4 and try_times < 100:
            try_times += 1
            corp_image = __corpImg()

        return Image.fromarray(corp_image)


    ''' 制造更多图片 '''
    def __getMoreImg(self, img_path):
        im_name = os.path.splitext(os.path.split(img_path)[1])[0]
        self.__calProgress(im_name, self.NUM_TRANSFORM + self.NUM_BLOCK_IMAGE + self.NUM_CORP_IMAGE)

        image = Image.open(img_path)

        # 保存原图
        file_no = 0
        image.save(os.path.join(self.IMG_MORE_PATH, '%s_%d.jpg' % (im_name, file_no)))

        # 小图
        file_no += 1
        small_image = self.__getSmallPig(image)
        small_image.save(os.path.join(self.IMG_MORE_PATH, '%s_%d.jpg' % (im_name, file_no)))

        # 水平翻转
        file_no += 1
        flip_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        flip_image.save(os.path.join(self.IMG_MORE_PATH, '%s_%d.jpg' % (im_name, file_no)))

        # 垂直翻转
        file_no += 1
        flip_image = image.transpose(Image.FLIP_TOP_BOTTOM)
        flip_image.save(os.path.join(self.IMG_MORE_PATH, '%s_%d.jpg' % (im_name, file_no)))

        # 亮度
        brightness_up = 1 + random.random() * 0.7
        brightness_down = 1 - random.random() * 0.8
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
        color_up = 1 + random.random() * 0.7
        color_down = 1 - random.random() * 0.6
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
        contrast_up = 1 + random.random() * 0.5
        contrast_down = 1 - random.random() * 0.4
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
            block_image = Image.fromarray( self.__getBlockImg(copy.deepcopy(np_image)) )

            file_no += 1
            block_image.save(os.path.join(self.IMG_MORE_PATH, '%s_%d.jpg' % (im_name, file_no)))

        # 随机裁剪图片
        for i in range(self.NUM_CORP_IMAGE):
            corp_image = self.__randomCorp(np_image)
            file_no += 1
            corp_image.save(os.path.join(self.IMG_MORE_PATH, '%s_%d.jpg' % (im_name, file_no)))


    def __getBlockImg(self, block_image, times = 100):
        w, h, c = block_image.shape

        def __getImg():
            ratio = 1.0 / random.randint(7, 12)
            block_w = int(w * ratio)
            block_h = int(h * ratio)

            _half_block_w = int(block_w / 2)
            _half_block_h = int(block_h / 2)

            _block_center_x = random.randrange(_half_block_w, w - _half_block_w)
            _block_center_y = random.randrange(_half_block_h, h - _half_block_h)

            _block_content = block_image[_block_center_x - _half_block_w: _block_center_x + _half_block_w,
                            _block_center_y - _half_block_h: _block_center_y + _half_block_h, :]

            return _block_content, _block_center_x, _block_center_y, _half_block_w, _half_block_h

        block_content, block_center_x, block_center_y, half_block_w, half_block_h = __getImg()
        while self.__calPigRatio(block_content) < self.MIN_BLOCK_PIG_RATIO and times > 0:
            times -= 1
            block_content, block_center_x, block_center_y, half_block_w, half_block_h = __getImg()

        block_image[block_center_x - half_block_w: block_center_x + half_block_w,
        block_center_y - half_block_h: block_center_y + half_block_h, :] = np.array([0, 0, 0], np.int8)
        return block_image


    @staticmethod
    def __calPigRatio(im):
        w, h, c = im.shape
        im_size = w * h

        real_size = 0
        for i, val_i in enumerate(im):
            for j, val_j in enumerate(val_i):
                r = float(val_j[0])
                g = float(val_j[1])
                b = float(val_j[2])

                if (80 < r < 96 and 70 < g < 80 and 65 < b < 80) \
                        or (145 < r < 155 and 115 < g < 121 and 100 < b < 115) \
                        or (141 < r < 147 and 113 < g < 118 and 112 < b < 120) \
                        or (114 < r < 120 and 94 < g < 100 and 94 < b < 100) \
                        or (52 < r < 60 and 40 < g < 50 and 40 < b < 50) \
                        or (95 < r < 105 and 82 < g < 89 and 78 < b < 85) \
                        or (123 < r < 131 and 106 < g < 113 and 95 < b < 103) \
                        or b < 35 or g < 45 or r < 40 or r > 252 \
                        or g > 225 or b > 215 or g / b > 1.17 or g / b < 0.8 \
                        or r / g > 3.1 or 2 > r / g > 1.45 or r / g < 1.05:
                    pass
                else:
                    real_size += 1

        return float(real_size) / im_size


    ''' 获取进度 '''
    def __calProgress(self, img_name, increment = 1):
        self.__progressIndex += increment
        progress = float(self.__progressIndex) / self.__progressLen * 100
        self.echo('progress: %.2f%% \t processing: %s \t \r' % (progress, img_name), False)


    ''' 输出展示 '''
    @staticmethod
    def echo(msg, crlf=True):
        if crlf:
            print msg
        else:
            sys.stdout.write(msg)
            sys.stdout.flush()


    def run(self):
        self.__getAlreadyExistList()
        self.__getImgList()

        for i, img_path in enumerate(self.__imgList):
            self.__getMoreImg(img_path)

        self.echo('done')


o_img = Img()
o_img.run()
