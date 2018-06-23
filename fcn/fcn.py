#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import sys
import math
import numpy as np
from PIL import Image
import tensorflow as tf

if '2.7' in sys.version:
    import Queue as queue
else:
    import queue

# 将运行路径切换到当前文件所在路径
cur_dir_path = os.path.abspath(os.path.split(__file__)[0])
if cur_dir_path:
    os.chdir(cur_dir_path)
    sys.path.append(cur_dir_path)
    sys.path.append(os.path.split(cur_dir_path)[0])

import lib.base as base
import load_cell as load
import model.vgg as vgg

''' 全卷积神经网络 '''


class FCN(base.NN):
    MODEL_NAME = 'fcn'  # 模型的名称

    ''' 参数的配置 '''

    BATCH_SIZE = 4  # 迭代的 epoch 次数
    EPOCH_TIMES = 100  # 随机梯度下降的 batch 大小

    NUM_CHANNEL = 3  # 输入图片为 3 通道，彩色
    NUM_CLASSES = 2  # 输出的类别

    # 学习率的相关参数
    BASE_LEARNING_RATE = 0.01  # 初始 学习率
    DECAY_RATE = 0.05  # 学习率 的 下降速率

    # 防止 overfitting 相关参数
    REGULAR_BETA = 0.01  # 正则化的 beta 参数
    KEEP_PROB = 0.5  # dropout 的 keep_prob

    # early stop
    MAX_VAL_LOSS_INCR_TIMES = 20  # 校验集 val_loss 连续 100 次没有降低，则 early stop

    ''' 类的配置 '''

    SHOW_PROGRESS_FREQUENCY = 2  # 每 SHOW_PROGRESS_FREQUENCY 个 step show 一次进度 progress

    ''' 模型的配置；采用了 VGG16 模型的 FCN '''

    VGG_MODEL = vgg.VGG.load()  # 加载 VGG 模型

    MODEL = [
        {
            'name': 'conv1_1',
            'type': 'conv',
            'W': VGG_MODEL['conv1_1'][0],
            'b': VGG_MODEL['conv1_1'][1],
            'trainable': False,
        },
        {
            'name': 'conv1_2',
            'type': 'conv',
            'W': VGG_MODEL['conv1_2'][0],
            'b': VGG_MODEL['conv1_2'][1],
            'trainable': False,
        },
        {
            'name': 'pool_1',
            'type': 'pool',
            'k_size': 2,
            'pool_type': 'avg',
        },
        {
            'name': 'conv2_1',
            'type': 'conv',
            'W': VGG_MODEL['conv2_1'][0],
            'b': VGG_MODEL['conv2_1'][1],
            'trainable': False,
        },
        {
            'name': 'conv2_2',
            'type': 'conv',
            'W': VGG_MODEL['conv2_2'][0],
            'b': VGG_MODEL['conv2_2'][1],
            'trainable': False,
        },
        {
            'name': 'pool_2',
            'type': 'pool',
            'k_size': 2,
            'pool_type': 'avg',
        },
        {
            'name': 'conv3_1',
            'type': 'conv',
            'W': VGG_MODEL['conv3_1'][0],
            'b': VGG_MODEL['conv3_1'][1],
            'trainable': False,
        },
        {
            'name': 'conv3_2',
            'type': 'conv',
            'W': VGG_MODEL['conv3_2'][0],
            'b': VGG_MODEL['conv3_2'][1],
            'trainable': False,
        },
        {
            'name': 'conv3_3',
            'type': 'conv',
            'W': VGG_MODEL['conv3_3'][0],
            'b': VGG_MODEL['conv3_3'][1],
            'trainable': False,
        },
        {
            'name': 'pool_3',
            'type': 'pool',
            'k_size': 2,
            'pool_type': 'avg',
        },
        {
            'name': 'conv4_1',
            'type': 'conv',
            'W': VGG_MODEL['conv4_1'][0],
            'b': VGG_MODEL['conv4_1'][1],
            'trainable': False,
        },
        {
            'name': 'conv4_2',
            'type': 'conv',
            'W': VGG_MODEL['conv4_2'][0],
            'b': VGG_MODEL['conv4_2'][1],
            'trainable': False,
        },
        {
            'name': 'conv4_3',
            'type': 'conv',
            'W': VGG_MODEL['conv4_3'][0],
            'b': VGG_MODEL['conv4_3'][1],
            'trainable': False,
        },
        {
            'name': 'pool_4',
            'type': 'pool',
            'k_size': 2,
            'pool_type': 'avg',
        },
        {
            'name': 'conv5_1',
            'type': 'conv',
            'W': VGG_MODEL['conv5_1'][0],
            'b': VGG_MODEL['conv5_1'][1],
            'trainable': False,
        },
        {
            'name': 'conv5_2',
            'type': 'conv',
            'W': VGG_MODEL['conv5_2'][0],
            'b': VGG_MODEL['conv5_2'][1],
            'trainable': False,
        },
        {
            'name': 'conv5_3',
            'type': 'conv',
            'W': VGG_MODEL['conv5_3'][0],
            'b': VGG_MODEL['conv5_3'][1],
            'trainable': False,
        },
        {
            'name': 'pool_5',
            'type': 'pool',
            'k_size': 2,
            'pool_type': 'max',
        },
        {
            'name': 'conv6',
            'type': 'conv',
            'shape': [VGG_MODEL['conv5_3'][0].shape[3], 4096],
            'k_size': [7, 7],
        },
        {
            'name': 'dropout_6',
            'type': 'dropout',
        },
        {
            'name': 'conv7',
            'type': 'conv',
            'shape': [4096, 4096],
            'k_size': [1, 1],
        },
        {
            'name': 'dropout_7',
            'type': 'dropout',
        },
        {
            'name': 'conv8',
            'type': 'conv',
            'shape': [4096, NUM_CLASSES],
            'k_size': [1, 1],
            'activate': False,
        },
        {
            'name': 'tr_conv_1',
            'type': 'tr_conv',
            'shape': [VGG_MODEL['conv4_3'][0].shape[3], NUM_CLASSES],  # 对应 [ pool_4 层的 channel, NUM_CLASSES ]
            'k_size': [4, 4],
            'output_shape_index': 'pool_4',  # 对应 pool_4 层的 shape
        },
        {
            'name': 'add_1',
            'type': 'add',
            'layer_index': 'pool_4',  # 对应 pool_4 层
        },
        {
            'name': 'tr_conv_2',
            'type': 'tr_conv',
            'shape': [VGG_MODEL['conv3_3'][0].shape[3], VGG_MODEL['conv4_3'][0].shape[3]],
            # 对应 [ pool_3 层的 channel, pool_4 层的 channel ]
            'k_size': [4, 4],
            'output_shape_index': 'pool_3',  # 对应 pool_3 层的 shape
        },
        {
            'name': 'add_2',
            'type': 'add',
            'layer_index': 'pool_3',  # 对应 pool_3 层
        },
        {
            'name': 'tr_conv_3',
            'type': 'tr_conv',
            'shape': [NUM_CLASSES, VGG_MODEL['conv3_3'][0].shape[3]],
            'k_size': [16, 16],
            'stride': 8,
            'output_shape_x': [None, None, None, NUM_CLASSES],  # 对应输入层的 shape
        },
    ]

    ''' 自定义 初始化变量 过程 '''

    def init(self):
        # 加载数据
        self.load()

        # 常量
        self.__iter_per_epoch = int(self.__train_size // self.BATCH_SIZE)
        self.__steps = self.EPOCH_TIMES * self.__iter_per_epoch

        # 输入 与 label
        self.__image = tf.placeholder(tf.float32, [None, None, None, self.NUM_CHANNEL], name='X')
        self.__mask = tf.placeholder(tf.float32, [None, None, None, self.NUM_CLASSES], name='y')

        # dropout 的 keep_prob
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # 随训练次数增多而衰减的学习率
        self.__learning_rate = self.get_learning_rate(
            self.BASE_LEARNING_RATE, self.global_step, self.__steps, self.DECAY_RATE, staircase=False
        )

        self.__has_rebuild = False

    ''' 加载数据 '''

    def load(self):
        sort_list = load.Data.get_sort_list()
        self.__train_set = load.Data(0.0, 0.9, 'train', sort_list)
        self.__val_set = load.Data(0.9, 1.0, 'validation', sort_list)
        # self.__test_set = load.Data(0.8, 1.0, 'test', sort_list)

        self.__train_size = self.__train_set.get_size()
        self.__val_size = self.__val_set.get_size()
        # self.__test_size = self.__test_set.get_size()

    ''' 模型 '''

    def model(self):
        self.__output = self.parse_model(self.__image)
        self.__output_mask = tf.argmax(self.__output, axis=3, name="output_mask")

    ''' 重建模型 '''

    def rebuild_model(self):
        self.__output = self.parse_model_rebuild(self.__image)
        self.__output_mask = tf.argmax(self.__output, axis=3, name="output_mask")

    ''' 计算 loss '''

    def get_loss(self):
        with tf.name_scope('loss'):
            logits = tf.to_float(tf.reshape(self.__output, [-1, self.NUM_CLASSES]), name='logits')
            labels = tf.to_float(tf.reshape(self.__mask, [-1, self.NUM_CLASSES]), name='labels')

            self.__loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels), name='entropy'
            )

    ''' 将图片输出到 tensorboard '''

    def __summary(self):
        with tf.name_scope('summary'):
            mask = tf.argmax(self.__mask, axis=3)

            # 转换 mask 和 output_mask 的 shape 成为 列向量
            mask = tf.to_float(tf.expand_dims(mask, dim=3), name='truth_mask')
            output_mask = tf.to_float(tf.expand_dims(self.__output_mask, dim=3), name='output_mask')

            # 输出图片到 tensorboard
            tf.summary.image('input_image', self.__image, max_outputs=2)
            tf.summary.image('truth_mask', tf.cast(mask * self.__image, tf.uint8), max_outputs=2)
            tf.summary.image('output_image', tf.cast(output_mask * self.__image, tf.uint8), max_outputs=2)

            # 记录 loss 到 tensorboard
            self.__loss_placeholder = tf.placeholder(tf.float32, name='loss')
            tf.summary.scalar('mean_loss', self.__loss_placeholder)

    ''' 测量数据集的 loss '''

    def __measure_loss(self, data_set):
        times = int(math.ceil(float(data_set.get_size()) / self.BATCH_SIZE))

        mean_loss = 0
        for i in range(times):
            batch_x, batch_y = data_set.next_batch(self.BATCH_SIZE)
            feed_dict = {self.__image: batch_x, self.__mask: batch_y, self.keep_prob: 1.0}
            loss = self.sess.run(self.__loss, feed_dict)
            mean_loss += loss

            # progress = float(i + 1) / times * 100
            # self.echo('\r measuring loss progress: %.2f%% | %d \t' % (progress, times), False)

        return mean_loss / times

    ''' 将 输出的 mask 中不连续的点去掉，只留下质心周围连续的点 '''

    @staticmethod
    def __mask2img(mask, np_image):
        h, w = mask.shape

        data = []
        for i in range(h):
            for j in range(w):
                if mask[i, j] != 0:
                    data.append([i, j])

        data = np.array(data)
        center = np.cast['int32'](np.mean(data, axis=0))

        if mask[center[0], center[1]] == 0:
            dis_mat = np.sum(np.power(data - center, 2), axis=1)

            dis_list = []
            for i, dis in enumerate(dis_mat):
                dis_list.append([i, dis])
            dis_list.sort(key=lambda _x: _x[1])

            center = data[dis_list[0][0]]

        s = set()
        q = queue.Queue()
        q.put(center)

        while not q.empty():
            x, y = q.get()

            if x + 1 < h:
                c = (x + 1, y)
                if mask[c[0], c[1]] != 0:
                    if c not in s:
                        s.add(c)
                        q.put(c)

            if x - 1 >= 0:
                c = (x - 1, y)
                if mask[c[0], c[1]] != 0:
                    if c not in s:
                        s.add(c)
                        q.put(c)

            if y - 1 >= 0:
                c = (x, y - 1)
                if mask[c[0], c[1]] != 0:
                    if c not in s:
                        s.add(c)
                        q.put(c)

            if y + 1 < w:
                c = (x, y + 1)
                if mask[c[0], c[1]] != 0:
                    if c not in s:
                        s.add(c)
                        q.put(c)

        new_mask = np.zeros_like(mask)
        for c in s:
            new_mask[c[0], c[1]] = 1

        new_mask = np.expand_dims(new_mask, axis=2)
        return np.cast['uint8'](new_mask * np_image)

    ''' 主函数 '''

    def run(self):
        # 生成模型
        self.model()

        # 计算 loss
        self.get_loss()

        # 正则化
        # self.__loss = self.regularize_trainable(self.__loss, self.REGULAR_BETA)

        # 生成训练的 op
        train_op = self.get_train_op(self.__loss, self.__learning_rate, self.global_step)

        # tensorboard 相关记录
        self.__summary()

        # 初始化所有变量
        self.init_variables()

        # TensorBoard merge summary
        self.merge_summary()

        best_val_loss = 9.9999e12  # 校验集 loss 最好的情况
        increase_val_loss_times = 0  # 校验集 loss 连续上升次数
        mean_loss = 0

        self.echo('\nepoch:')

        for step in range(self.__steps):
            if step % self.SHOW_PROGRESS_FREQUENCY == 0:
                epoch_progress = float(step) % self.__iter_per_epoch / self.__iter_per_epoch * 100.0
                step_progress = float(step) / self.__steps * 100.0
                self.echo('\rstep: %d (%d|%.2f%%) / %d|%.2f%% \t\t' % (step, self.__iter_per_epoch, epoch_progress,
                                                                       self.__steps, step_progress), False)

            batch_x, batch_y = self.__train_set.next_batch(self.BATCH_SIZE)

            import matplotlib.pyplot as plt
            plt.imshow(np.cast['uint8'](batch_y[0, :, :, 1] * 255))
            plt.show()
            exit()

            feed_dict = {self.__image: batch_x, self.__mask: batch_y,
                         self.keep_prob: self.KEEP_PROB}
            _, train_loss = self.sess.run([train_op, self.__loss], feed_dict)

            mean_loss += train_loss

            if step % self.__iter_per_epoch == 0 and step != 0:
                epoch = int(step // self.__iter_per_epoch)

                feed_dict[self.__loss_placeholder] = mean_loss / self.__iter_per_epoch
                mean_loss = 0
                self.add_summary_train(feed_dict, epoch)

                # 测试 校验集 的 loss
                mean_val_loss = self.__measure_loss(self.__val_set)
                batch_val_x, batch_val_y = self.__val_set.next_batch(self.BATCH_SIZE)
                feed_dict = {self.__image: batch_val_x, self.__mask: batch_val_y, self.keep_prob: 1.0,
                             self.__loss_placeholder: mean_val_loss}
                self.add_summary_val(feed_dict, epoch)

                if best_val_loss > mean_val_loss:
                    best_val_loss = mean_val_loss
                    increase_val_loss_times = 0

                    self.echo('\n best_val_loss: %.2f \t ' % best_val_loss)
                    self.save_model_w_b()

                else:
                    increase_val_loss_times += 1
                    if increase_val_loss_times > self.MAX_VAL_LOSS_INCR_TIMES:
                        break

        self.close_summary()  # 关闭 TensorBoard

        self.restore_model_w_b()  # 恢复模型
        self.rebuild_model()  # 重建模型
        self.get_loss()  # 重新 get loss

        self.init_variables()  # 重新初始化变量

        train_loss = self.__measure_loss(self.__train_set)
        val_loss = self.__measure_loss(self.__val_set)
        # test_loss = self.__measure_loss(self.__test_set)

        self.echo('\ntrain mean loss: %.6f' % train_loss)
        self.echo('validation mean loss: %.6f' % val_loss)
        # self.echo('test mean loss: %.6f' % test_loss)

        self.echo('\ndone')

        # show some val image result
        batch_x, batch_y = self.__val_set.next_batch(self.BATCH_SIZE)
        feed_dict = {self.__image: batch_x, self.keep_prob: 1.0}
        output_mask = self.sess.run(self.__output_mask, feed_dict)

        output_mask = np.expand_dims(output_mask, axis=3)
        for i in range(3):
            mask = output_mask[i]
            image = batch_x[i]
            new_image = np.cast['uint8'](mask * image)

            o_image = Image.fromarray(np.cast['uint8'](image))
            o_image.show()

            o_new_image = Image.fromarray(new_image)
            o_new_image.show()

    def use_model(self, np_image):
        if not self.__has_rebuild:
            self.restore_model_w_b()  # 恢复模型
            self.rebuild_model()  # 重建模型

            self.init_variables()  # 初始化所有变量
            self.__has_rebuild = True

        np_image = np.expand_dims(np_image, axis=0)
        feed_dict = {self.__image: np_image, self.keep_prob: 1.0}
        output_mask = self.sess.run(self.__output_mask, feed_dict)

        return self.__mask2img(output_mask[0], np_image[0])  # 将 mask 待人 image 并去掉外部的点点

    def test_model(self):
        self.restore_model_w_b()  # 恢复模型
        self.rebuild_model()  # 重建模型
        self.get_loss()  # 重新 get loss

        self.init_variables()  # 初始化所有变量

        train_loss = self.__measure_loss(self.__train_set)
        val_loss = self.__measure_loss(self.__val_set)
        # test_loss = self.__measure_loss(self.__test_set)

        self.echo('\ntrain mean loss: %.6f' % train_loss)
        self.echo('validation mean loss: %.6f' % val_loss)
        # self.echo('test mean loss: %.6f' % test_loss)

        self.echo('\ndone')

        # show some val image result
        batch_x, batch_y = self.__val_set.next_batch(self.BATCH_SIZE)
        feed_dict = {self.__image: batch_x, self.keep_prob: 1.0}
        output_mask = self.sess.run(self.__output_mask, feed_dict)

        output_mask = np.expand_dims(output_mask, axis=3)
        for i in range(3):
            mask = output_mask[i]
            image = batch_x[i]
            new_image = np.cast['uint8'](mask * image)

            o_image = Image.fromarray(np.cast['uint8'](image))
            o_image.show()

            o_new_image = Image.fromarray(new_image)
            o_new_image.show()

o_fcn = FCN()
o_fcn.run()
# o_fcn.test_model()
