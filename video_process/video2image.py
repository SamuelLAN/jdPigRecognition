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


'''
 将 视频 转换为 图片，并做 data argument
'''
class Transformer:
    VIDEO_PATH = r'Data/TrainVideo'
    IMG_PATH = r'Data/TrainImg'

    IMG_PER_FRAME = 29

    def __init__(self):
        self.__videoList = []   # 存放视频的路径


    # 获取视频的路径列表
    def __getVideoList(self):
        for file_name in os.listdir(self.VIDEO_PATH):
            if os.path.splitext(file_name)[1] != '.mp4':
                continue

            video_path = os.path.join(self.VIDEO_PATH, file_name)   # 获取每个视频的路径
            self.__videoList.append(video_path)


    # 根据 video_path 读取视频，并每 IMG_PER_FRAME 帧保存一幅图片
    def __getImage(self, video_path):
        video_no = os.path.splitext( os.path.split(video_path)[1] )[0]  # video 编号
        file_no = 1                         # video 对应的图片的 no

        cap = cv2.VideoCapture(video_path)  # 初始化 Video
        if not cap.isOpened():              # 检查 video 是否打开
            cap.open()

        ret, _ = cap.read()     # 因为第一次 cap.read() 返回的 ret 为 False
        ret = True              # 重置 ret 为 True

        count = 0
        while ret:
            ret, frame = cap.read() # 读取视频
            count += 1

            if count % self.IMG_PER_FRAME == 0:     # 每 IMG_PER_FRAME 帧保存一次图片
                img_path = os.path.join(self.IMG_PATH, '%s_%d.jpg' % (video_no, file_no))
                cv2.imwrite(img_path, frame)
                file_no += 1

                self.echo('save %s' % img_path)

            cv2.waitKey(1)

        cap.release()           # 释放视频


    ''' 输出展示 '''
    @staticmethod
    def echo(msg, crlf=True):
        if crlf:
            print msg
        else:
            sys.stdout.write(msg)
            sys.stdout.flush()


    # 主函数
    def run(self):
        self.__getVideoList()

        video_len = len(self.__videoList)

        for i, video_path in enumerate(self.__videoList):
            progress = float(i) / video_len * 100
            self.echo('progress: %.2f \t transforming %s \t \r' % (progress, video_path))

            self.__getImage(video_path)

        self.echo('\ndone')


o_transformer = Transformer()
o_transformer.run()
