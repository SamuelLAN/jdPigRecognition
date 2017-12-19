#!/usr/bin/Python
# -*- coding: utf-8 -*-
from __future__ import print_function
import tensorflow as tf
from math import sqrt
from numpy import hstack
import re
import sys
import os
import time
import platform
import traceback
import numpy as np
from multiprocessing import Process
from six.moves import cPickle as pickle
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

'''
 网络的基类
 提供了：
    基类默认初始化的操作
      子类请不要重载构造函数
      要初始化变量可以通过 重载 self.init 函数
    子类需要实现的接口
      init 初始化各种变量
      load 加载数据
      model 重载模型
      inference 前向传导 (若想使用 TensorBoard 训练时同时看到 训练集、测试集 的准确率，需要实现 inference)
      get_loss loss 函数
      run 主函数
    初始化变量
      权重矩阵的初始化
      偏置量的初始化
      学习率的初始化
      执行 tf 的所有变量的初始化
    保存模型
    TensorBoard summary
    常用模型
      全连接模型
      深度模型 CNN
    与 模型有关 的 常用函数
      计算 h
      激活函数
      卷积函数
      pooling 函数
    与 训练有关 的 常用函数
      get train op
    防止 overfitting 的 trick
      regularize
      dropout
    他常用函数 (与 DL 无关)
      echo

 public 常量：
 
 public 变量：
 
 private 变量：

'''


class NN:
    MODEL_NAME = 'model_name'  # 模型的名称

    ''' 参数的配置 '''

    EPOCH_TIMES = 100  # 迭代的 epoch 次数
    BATCH_SIZE = 100  # 随机梯度下降的 batch 大小

    # 学习率的相关参数
    BASE_LEARNING_RATE = 0.1  # 初始 学习率
    DECAY_RATE = 0.95  # 学习率 的 下降速率

    # batch_normalize 的相关参数
    BN_DECAY = 0.9997  # batch normalize 中移动平均的下降率
    BN_EPSILON = 0.001  # 给 std 加上的一个极小的数值，避免除数为 0

    # dropout 相关的参数
    KEEP_PROB = 0.5  # dropout 的 keep_prob；若 KEEP_PROB_DICT 为空，使用 KEEP_PROB
    KEEP_PROB_DICT = {}  # 若该变量不为空，则 dropout 使用该变量；否则使用 KEEP_PROB

    CONV_WEIGHT_STDDEV = 0.1  # truncated normal distribution 的 std

    EPSILON = 0.00001   # 输入 做 batch normalize 时需要用到

    ''' 模型的配置 '''

    MODEL = []  # 深度模型的配置

    ''' 类的配置 '''

    USE_MULTI = False  # 是否需要训练多个网络；默认为 false
    USE_BN = False  # 网络里是否使用了 batch normalize
    USE_BN_INPUT = False  # 输入是否使用 batch normalize

    TENSORBOARD_SHOW_IMAGE = False  # 默认不将 image 显示到 TensorBoard，以免影响性能
    TENSORBOARD_SHOW_GRAD = False  # 默认不将 gradient 显示到 TensorBoard，以免影响性能
    TENSORBOARD_SHOW_ACTIVATION = False  # 默认不将 activation 显示到 TensorBoard，以免影响性能

    ''' collection 的名字 '''

    VARIABLE_COLLECTION = 'variables'
    UPDATE_OPS_COLLECTION = 'update_ops'

    # ******************************** 基类默认初始化的操作 ****************************

    def __init__(self):
        self.tbProcess = None  # tensorboard process
        self.__init()  # 执行基类的初始化函数

    ''' 析构函数 '''

    def __del__(self):
        # 判断 tensorboard 是否已经结束；若无, kill it
        NN.kill_tensorboard_if_running()
        if not isinstance(self.tbProcess, type(None)):
            self.tbProcess.join(10)
            self.tbProcess.terminate()

    ''' 初始化 '''

    def __init(self):
        # self.net = []  # 存放每层网络的 feature map
        # self.w_list = []  # 存放权重矩阵的 list
        # self.b_list = []  # 存放偏置量的 list

        self.net = {}  # 存放每层网络的 feature map
        self.w_dict = {}  # 存放权重矩阵的 dict
        self.b_dict = {}  # 存放偏置量的 dict

        # 只有 self.USE_MULTI == True 时才需要用到
        self.multi_net = []  # 存放多个网络的 每层网络的 feature map
        self.multi_w_dict = []  # 存放多个网络的 权重矩阵的 list
        self.multi_b_dict = []  # 存放多个网络的 偏置量的 list

        # batch normalize 参数存放的变量
        self.__beta_dict = {}
        self.__gamma_dict = {}
        self.__moving_mean_dict = {}
        self.__moving_std_dict = {}

        # 给 input batch normalize 使用
        self.mean_x = 0
        self.std_x = 0.0001

        # 若 USE_MULTI 为 True 时，该值才有意义
        self.net_id = 0

        # dropout 的 keep_prob，为 tensor 对象
        self.keep_prob = None
        self.keep_prob_dict = {}

        # tensor is_train，用于 batch_normalize; 没有用 bn 时，无需加入 feed_dict
        self.t_is_train = tf.placeholder(tf.bool, name='is_train')

        # 程序运行的开始时间；用于 get_model_path 和 get_summary_path 时使用
        self.__start_time = time.strftime('%Y_%m_%d_%H_%M_%S')

        # 初始化 model 路径
        self.__model_path = ''
        self.get_model_path()  # 生成存放模型的文件夹 与 路径

        # 初始化 tensorboard summary 的文件夹路径 并 开启 tensorboard
        self.__summaryPath = ''
        self.get_summary_path()

        # merge summary 时需要用到；判断是否已经初始化 summary writer
        self.__init_summary_writer = False

        # 执行定制化的 初始化操作；每个子类都需要重载该函数
        self.init()

        ''' 若只需训练一个网络 '''
        if not self.USE_MULTI:
            self.global_step = self.get_global_step()  # 记录全局训练状态的 global step
            self.sess = tf.Session()  # 初始化 sess

            # tensorflow 自带的保存网络的方法；目前采用了自己实现的保存网络的方法
            # self.saver = tf.train.Saver()                 # 初始化 saver; 用于之后保存 网络结构

    # ******************************* 子类需要实现的接口 *******************************

    ''' 初始化各种 变量 常量 '''

    def init(self):
        pass

    ''' 加载数据 '''

    def load(self):
        pass

    ''' 模型 '''

    def model(self):
        pass

    ''' 前向推导 '''

    def inference(self):
        pass

    ''' 计算 loss '''

    def get_loss(self):
        pass

    ''' 主函数 '''

    def run(self):
        pass

    # *************************** 初始化变量 ****************************

    ''' 初始化所有变量 '''

    def init_variables(self):
        self.sess.run(tf.global_variables_initializer())

    ''' 初始化权重矩阵 '''

    @staticmethod
    def init_weight(shape):
        return tf.Variable(
            tf.truncated_normal(
                shape,
                stddev=1.0 / NN.CONV_WEIGHT_STDDEV,
            ),
            name='weight'
        )

    ''' 初始化权重矩阵 '''

    @staticmethod
    def init_weight_w(w, trainable=True):
        return tf.Variable(w, trainable=trainable, name='weight')

    ''' 初始化 bias '''

    @staticmethod
    def init_bias(shape):
        if len(shape) == 4:
            nodes = shape[2]
        else:
            nodes = shape[-1]

        return tf.Variable(tf.zeros([nodes]), name='bias')

    ''' 初始化 bias '''

    @staticmethod
    def init_bias_b(b, trainable=True):
        return tf.Variable(b, trainable=trainable, name='bias')

    # def get_variable(self, name, shape, initializer, weight_decay=0.0, dtype='float', trainable=True):
    #     with tf.variable_scope('regularizer'):
    #         if weight_decay > 0:
    #             regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    #         else:
    #             regularizer = None
    #     collections = [tf.GraphKeys.GLOBAL_VARIABLES, self.VARIABLE_COLLECTION]
    #     return tf.get_variable(name, shape=shape, initializer=initializer, dtype=dtype,
    #                            regularizer=regularizer, collections=collections, trainable=trainable)

    ''' 获取全局的训练 step '''

    @staticmethod
    def get_global_step():
        return tf.Variable(0, name='global_step', trainable=False)

    ''' 获取随迭代次数下降的学习率 '''

    @staticmethod
    def get_learning_rate(base_learning_rate, cur_step, decay_times, decay_rate=0.95, staircase=False,
                          tensorboard=False):
        learning_rate = tf.train.exponential_decay(
            base_learning_rate,  # Base learning rate.
            cur_step,  # Current index into the dataset.
            decay_times,  # Decay step.
            decay_rate,  # Decay rate.
            staircase=staircase
        )
        if tensorboard:
            tf.summary.scalar('learning_rate', learning_rate)
        return learning_rate

    # *************************** 保存模型 ***************************

    # ''' 恢复模型 '''
    #
    # def restore_old_model(self):
    #     ckpt = tf.train.get_checkpoint_state(os.path.split(self.get_model_path())[1])
    #     if ckpt and ckpt.model_checkpoint_path:
    #         self.saver.restore(self.sess, ckpt.model_checkpoint_path)
    #         self.echo('Model restored ...')

    # ''' 保存模型 '''
    #
    # def save_model(self):
    #     self.saver.save(self.sess, self.get_model_path() + '.ckpt', self.global_step)
    #     # self.saver.save(self.sess, self.get_model_path())

    # ''' 恢复模型 '''
    #
    # def restore_model(self):
    #     model_path = self.get_model_path()
    #     self.saver = tf.train.import_meta_graph('%s.meta' % model_path)
    #     self.saver.restore(self.sess, tf.train.latest_checkpoint(os.path.split(model_path)[0]))
    #     self.graph = self.sess.graph

    ''' 自己实现的 save model '''

    def save_model_w_b(self):
        if self.USE_MULTI:
            model_path = '%s_%d.pkl' % (self.get_model_path(), self.net_id)
            self_w_dict = self.multi_w_dict[self.net_id]
            self_b_dict = self.multi_b_dict[self.net_id]
        else:
            model_path = self.get_model_path()
            self_w_dict = self.w_dict
            self_b_dict = self.b_dict

        model_name = os.path.split(model_path)[1]
        self.echo('\nSaving model to %s ...' % model_name)

        w_dict = {}
        for name_scope, w in self_w_dict.items():
            name = w.name.split(':')[0]
            w_dict[name_scope] = [name, self.sess.run(w)]

        b_dict = {}
        for name_scope, b in self_b_dict.items():
            name = b.name.split(':')[0]
            b_dict[name_scope] = [name, self.sess.run(b)]

        save_dict = {
            'w_dict': w_dict,
            'b_dict': b_dict,
        }

        # 若有使用 batch normalize
        if self.USE_BN:
            beta_dict = {}
            for name_scope, tensor in self.__beta_dict.items():
                beta_dict[name_scope] = self.sess.run(tensor)

            gamma_dict = {}
            for name_scope, tensor in self.__gamma_dict.items():
                gamma_dict[name_scope] = self.sess.run(tensor)

            moving_mean_dict = {}
            for name_scope, tensor in self.__moving_mean_dict.items():
                moving_mean_dict[name_scope] = self.sess.run(tensor)

            moving_std_dict = {}
            for name_scope, tensor in self.__moving_std_dict.items():
                moving_std_dict[name_scope] = self.sess.run(tensor)

            save_dict['beta'] = beta_dict
            save_dict['gamma'] = gamma_dict
            save_dict['moving_mean'] = moving_mean_dict
            save_dict['moving_std'] = moving_std_dict

        # 若 input 有使用 batch normalize
        if self.USE_BN_INPUT:
            save_dict['mean_x'] = self.mean_x
            save_dict['std_x'] = self.std_x

        with open(model_path, 'wb') as f:
            pickle.dump(save_dict, f, 2)

        self.echo('Finish saving model ')

    def restore_model_w_b(self):
        if self.USE_MULTI:
            model_path = '%s_%d.pkl' % (self.get_model_path(), self.net_id)
            self.multi_w_dict[self.net_id] = {}
            self.multi_b_dict[self.net_id] = {}
            w_dict = self.multi_w_dict[self.net_id]
            b_dict = self.multi_b_dict[self.net_id]
        else:
            model_path = self.get_model_path()
            self.w_dict = {}
            self.b_dict = {}
            w_dict = self.w_dict
            b_dict = self.b_dict

        model_name = os.path.split(model_path)[1]
        self.echo('\nRestoring model to %s ...' % model_name)

        with open(model_path, 'rb') as f:
            save_dict = pickle.load(f)

        for name_scope, (name, w_value) in save_dict['w_dict'].items():
            w_dict[name_scope] = tf.Variable(w_value, trainable=False, name=name)

        for name_scope, (name, w_value) in save_dict['b_dict'].items():
            b_dict[name_scope] = tf.Variable(w_value, trainable=False, name=name)

        if self.USE_BN:
            self.__beta_dict = {}
            self.__gamma_dict = {}
            self.__moving_mean_dict = {}
            self.__moving_std_dict = {}

            for name_scope, value in save_dict['beta'].items():
                self.__beta_dict[name_scope] = tf.Variable(value, trainable=False, name='%s/beta' % name_scope)

            for name_scope, value in save_dict['gamma'].items():
                self.__gamma_dict[name_scope] = tf.Variable(value, trainable=False, name='%s/gamma' % name_scope)

            for name_scope, value in save_dict['moving_mean'].items():
                self.__moving_mean_dict[name_scope] = tf.Variable(value, trainable=False, name='%s/moving_mean' % name_scope)

            for name_scope, value in save_dict['moving_std'].items():
                self.__moving_std_dict[name_scope] = tf.Variable(value, trainable=False, name='%s/moving_variance' % name_scope)

        if self.USE_BN_INPUT:
            self.mean_x = save_dict['mean_x']
            self.std_x = save_dict['std_x']

        self.echo('Finish restoring ')

    ''' 根据 name 获取 tensor 变量 '''

    def get_variable_by_name(self, name):
        return self.graph.get_tensor_by_name(name)

    ''' 获取存放模型的路径 '''

    def get_model_path(self):
        if self.__model_path:
            return self.__model_path

        cur_dir = os.path.split(os.path.abspath(__file__))[0]
        model_dir = os.path.join(cur_dir, 'model')

        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        model_dir = os.path.join(model_dir, self.MODEL_NAME.split('.')[0])
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        self.__model_path = os.path.join(model_dir, '%s_%s' % (self.MODEL_NAME, self.__start_time))
        return self.__model_path

    ''' 设置模型的路径 '''

    def set_model_path(self, path):
        self.__model_path = path

    # ************************** TensorBoard summary ************************

    ''' TensorBoard merge summary '''

    def merge_summary(self):
        self.__mergedSummaryOp = tf.summary.merge_all()

        if not self.__init_summary_writer:
            if tf.gfile.Exists(self.__summaryPath):
                tf.gfile.DeleteRecursively(self.__summaryPath)
            self.__summaryWriterTrain = tf.summary.FileWriter(
                os.path.join(self.__summaryPath, 'train'), self.sess.graph)
            self.__summaryWriterVal = tf.summary.FileWriter(
                os.path.join(self.__summaryPath, 'validation'), self.sess.graph)
            self.__init_summary_writer = True

    ''' TensorBoard add sumary training '''

    def add_summary_train(self, feed_dict, step):
        summary_str = self.sess.run(self.__mergedSummaryOp, feed_dict)
        self.__summaryWriterTrain.add_summary(summary_str, step)
        self.__summaryWriterTrain.flush()

    ''' TensorBoard add sumary validation '''

    def add_summary_val(self, feed_dict, step):
        summary_str = self.sess.run(self.__mergedSummaryOp, feed_dict)
        self.__summaryWriterVal.add_summary(summary_str, step)
        self.__summaryWriterVal.flush()

    ''' TensorBoard close '''

    def close_summary(self):
        self.__summaryWriterTrain.close()
        self.__summaryWriterVal.close()

    ''' 输出前 num 个节点的图像到 TensorBoard '''

    def image_summary(self, tensor_4d, num, name):
        with tf.name_scope('image_summary'):
            index = self.global_step % self.BATCH_SIZE  # 让每次输出不同图片，取不同 index 的图片
            shape = list(hstack(
                [-1, [int(j) for j in tensor_4d.shape[1: 3]], -1]))  # 生成的 shape 为 [-1, image_width, image_height, 1]
            image = tf.concat([tensor_4d[index, :, :, :] for j in range(num)], 0)  # 将多幅图像合并在一起

            # 必须 reshape 成 [?, image_size, image_size, 1 或 3 或 4]
            image = tf.reshape(image, shape)
            tf.summary.image(name, image)

    @staticmethod
    def activation_summary(var):
        if not isinstance(var, type(None)):
            with tf.name_scope('activation'):
                tf.summary.histogram(var.op.name + '/activation', var)
                tf.summary.scalar(var.op.name + '/sparsity', tf.nn.zero_fraction(var))

    @staticmethod
    def gradient_summary(grad, var):
        if not isinstance(grad, type(None)):
            with tf.name_scope('gradient'):
                tf.summary.histogram(var.op.name + 'gradient', grad)

    ''' 获取 summary path '''

    def get_summary_path(self):
        cur_dir = os.path.split(os.path.abspath(__file__))[0]
        summary_dir = os.path.join(cur_dir, 'summary')

        if not os.path.isdir(summary_dir):
            os.mkdir(summary_dir)

        summary_dir = os.path.join(summary_dir, self.MODEL_NAME.split('.')[0])
        if not os.path.isdir(summary_dir):
            os.mkdir(summary_dir)

        summary_dir = os.path.join(summary_dir, self.__start_time)
        if not os.path.isdir(summary_dir):
            os.mkdir(summary_dir)

        # 若是训练多个网络
        if self.USE_MULTI:
            summary_dir = os.path.join(summary_dir, 'net_%d' % self.net_id)
            if not os.path.isdir(summary_dir):
                os.mkdir(summary_dir)

        dirs = ['train', 'validation']
        for dir_name in dirs:
            dir_path = os.path.join(summary_dir, dir_name)
            if not os.path.isdir(dir_path):
                os.mkdir(dir_path)
            else:
                self.__remove_file_recursive(dir_path)

        self.__summaryPath = summary_dir

        # 异步在终端运行 tensorboard
        self.run_tensorboard(self.__summaryPath)
        return self.__summaryPath

    ''' 递归删除目录 '''

    @staticmethod
    def __remove_file_recursive(dir_path):
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

    @staticmethod
    def cmd(command):
        return os.popen(command).read()

    ''' 若 tensorboard 正在运行, kill tensorboard 进程 '''

    @staticmethod
    def kill_tensorboard_if_running():
        try:
            # 检查 tensorboard 是否正在运行
            if platform.system().lower() == 'windows':
                command = r'tasklist | findstr tensorboard'
                kill_command = r'taskkill /pid %d /F'
            else:
                command = r'ps aux | grep tensorboard'
                kill_command = r'kill -9 %d'

            ps_cmd = NN.cmd(command).replace('\r', '').split('\n')

            reg = re.compile(r'\tensorboard\s+', re.IGNORECASE)
            reg_space = re.compile(r'\s+')

            for line in ps_cmd:
                # 若 tensorboard 正在运行, kill 进程
                if reg.search(line):
                    pid = int(reg_space.split(line)[1])
                    NN.cmd(kill_command % pid)

        except Exception as ex:
            NN.echo('Error func: kill_tensorboard_if_running\n%s' % str(ex))

    '''
     同步状态 自动在终端打开 cmd (默认端口为 6006，port 参数可以自己指定端口)
    '''

    @staticmethod
    def __run_tensorboard_sync(path, port=6006):
        try:
            env = 'python27' if '2.7' in sys.version else 'tensor3.5'  # python 所在的 conda 环境
            prefix = 'source' if platform.system().lower() != 'windows' else ''  # 判断是什么操作系统

            # 若无需激活 python 环境，可把前面的 activate 相关语句删除
            NN.cmd('%s activate %s;tensorboard --logdir=%s --port=%d' % (prefix, env, path, port))

        except Exception as ex:
            NN.echo('Error func: __run_tensorboard_sync\n%s' % ex)

    ''' 异步状态，自动在终端打开 cmd (默认端口为 6006，port 参数可以自己指定端口) '''

    def run_tensorboard(self, path, port=6006):
        NN.kill_tensorboard_if_running()
        self.tbProcess = Process(target=NN.__run_tensorboard_sync, args=(path, port))
        self.tbProcess.start()

    # **************************** 常用模型 ***************************

    ''' 
      深度模型
      在 self.MODEL 中配置即可
        self.MODEL 为 list，存放的元素为按顺序依次往下的网络的层
      配置支持的 type:
        conv：卷积
            for example:
            {
                'name': 'conv_3',   # name_scope，默认为 'type_第i层'; 所有 type 都有 name，下面不再详写
                'type': 'conv',
                'shape': [NUM_CHANNEL, 32], # 若有 'W' 和 'b'，可以没有该值
                'k_size': [5, 5],   # 若有 'W'，可以没有该值
                'activate': True,   # 默认为 True
                'W': W,             # kernel；若没有该值，会自动根据 k_size 以及 shape 初始化
                'b': b,             # bias; 若没有该值，会自动根据 shape 初始化,
                'bn': True,         # batch_normalize 默认为 False
                'padding': 'VALID', # conv 的 padding，默认为 'SAME'; 只支持 'VALID' 或 'SAME'
                'stride': 2,        # 默认为 1
            }
        tr_conv: 反卷积(上采样)
            for example:
            {
                'type': 'tr_conv',
                'shape': [NUM_CHANNEL, 32],
                'k_size': [16, 16],         # 一般为 stride 的两倍
                'output_shape': [1, 256, 256, NUM_CHANNEL], # 若指定了 output_shape_index 或 output_shape_x，
                                                            # 无需设置该项；该项可空
                'output_shape_index': 'conv_2',    # output_shape 跟 conv_2 层网络的 shape 一致；层数必须低于当前层
                                            # 若指定了 output_shape 或 output_shape_x，无需设置该项；该项可空
                'output_shape_x': [None, None, None, NUM_CHANNEL] # 使用输入层的 shape 作为 output_shape, 
                                            # 并且在此基础上，根据提供的 'output_shape_x' 更改 shape (若元素不为 None)
                                            # 若指定了 output_shape 或 output_shape_index，无需设置该项；该项可空
                'stride': 8,                # 默认为 2；stride 相当于上采样的倍数
            }
        pool: 池化
            for example:
            {
                'type': 'pool',
                'k_size': 2,
                'stride': 2,            # 默认等于 k_size
                'pool_type': 'max',     # 只支持 'max' 或 'avg'; 若没有设置该项，默认为 max_pool
            }
        fc: 全连接
            for example:
            {
                'type': 'fc',
                'shape': [1024, NUM_CLASSES]
                'activate': False,
            }
        dropout: dropout
            for example:
            {
                'type': 'dropout',
                'keep_prob': 0.5,
            }
        add: 将上一层网络的输出 与 第 layer_index 层网络 sum
            for example:
            {
                'type': 'add',
                'layer_index': 'pool_3', # layer_index 是层的 name, 且层数必须小于当前
            }

    '''

    def parse_model(self, X):
        self.echo('\nStart building model ... ')

        if self.USE_MULTI:
            w_dict = {}
            b_dict = {}
            net = {}
        else:
            w_dict = self.w_dict
            b_dict = self.b_dict
            net = self.net

        a = X
        model_len = len(self.MODEL)
        for i, config in enumerate(self.MODEL):
            _type = config['type'].lower()
            name = '%s_%d' % (_type, i + 1) if 'name' not in config else config['name']

            # 卷积层
            if _type == 'conv':
                with tf.name_scope(name):
                    # 初始化变量
                    trainable = True if 'trainable' not in config or config['trainable'] else False
                    W = self.init_weight(config['k_size'] + config['shape']) \
                        if not 'W' in config else self.init_weight_w(config['W'], trainable)
                    b = self.init_bias(config['shape']) \
                        if not 'b' in config else self.init_bias_b(config['b'], trainable)
                    w_dict[name] = W
                    b_dict[name] = b

                    # 具体操作
                    stride = config['stride'] if 'stride' in config else 1
                    padding = 'SAME' if 'padding' not in config or config['padding'] == 'SAME' else 'VALID'

                    a = tf.add(self.conv2d(a, W, stride, padding), b)

                    if 'bn' in config and config['bn']:
                        a = self.batch_normal(a, self.t_is_train, name)

                    if not 'activate' in config or config['activate']:
                        a = self.activate(a)

                        if self.TENSORBOARD_SHOW_ACTIVATION:
                            self.activation_summary(a)

                    if self.TENSORBOARD_SHOW_IMAGE:
                        self.image_summary(a, 3, name)

            # 反卷积层 (上采样 transpose conv)
            elif _type == 'tr_conv':
                with tf.name_scope(name):
                    # 初始化变量
                    trainable = True if 'trainable' not in config or config['trainable'] else False
                    W = self.init_weight(config['k_size'] + config['shape']) \
                        if not 'W' in config else self.init_weight_w(config['W'], trainable)
                    b = self.init_bias(config['shape'][:-2] + [config['shape'][-1], config['shape'][-2]]) \
                        if not 'b' in config else self.init_bias_b(config['b'], trainable)
                    w_dict[name] = W
                    b_dict[name] = b

                    # 具体操作
                    if 'output_shape' in config:
                        output_shape = config['output_shape']
                    elif 'output_shape_index' in config:
                        output_shape = tf.shape(net[config['output_shape_index']])
                    elif 'output_shape_x' in config:
                        output_shape = tf.shape(X)
                        for j, val_j in enumerate(config['output_shape_x']):
                            if not val_j:
                                continue
                            tmp = tf.Variable([1 if k != j else 0 for k in range(4)], tf.int8)
                            output_shape *= tmp
                            tmp = tf.Variable([0 if k != j else val_j for k in range(4)], tf.int8)
                            output_shape += tmp
                    else:
                        output_shape = None

                    stride = config['stride'] if 'stride' in config else 2
                    a = self.conv2d_transpose_stride(a, W, b, output_shape, stride)

                    if self.TENSORBOARD_SHOW_IMAGE:
                        self.image_summary(a, 3, name)

            # 全连接层
            elif _type == 'fc':
                with tf.name_scope(name):
                    # 初始化变量
                    trainable = True if 'trainable' not in config or config['trainable'] else False
                    W = self.init_weight(config['shape']) if not 'W' in config \
                        else self.init_weight_w(config['W'], trainable)
                    b = self.init_bias(config['shape']) if not 'b' in config \
                        else self.init_bias_b(config['b'], trainable)
                    w_dict[name] = W
                    b_dict[name] = b

                    x = tf.reshape(a, [-1, config['shape'][0]])
                    a = tf.add(tf.matmul(x, W), b)

                    if ('activate' not in config and i < model_len - 1) or config['activate']:
                        a = self.activate(a)

            # 池化层
            elif _type == 'pool':
                with tf.name_scope(name):
                    k_size = [config['k_size'], config['k_size']]
                    stride = config['stride'] if 'stride' in config else None
                    if 'pool_type' not in config or config['pool_type'] == 'max':
                        a = self.max_pool(a, k_size, stride)
                    else:
                        a = self.avg_pool(a, k_size, stride)

            # 训练的 dropout
            elif _type == 'dropout':
                with tf.name_scope(name):
                    t_dropout = self.keep_prob_dict[name] if name in self.keep_prob_dict else self.keep_prob
                    a = self.dropout(a, t_dropout)

            # 将上一层的输出 与 第 layer_index 层的网络相加
            elif _type == 'add':
                with tf.name_scope(name):
                    a = tf.add(a, net[config['layer_index']])

            net[name] = a

        if self.USE_MULTI:
            self.multi_w_dict.append(w_dict)
            self.multi_b_dict.append(b_dict)
            self.multi_net.append(net)

        self.echo('Finish building model ')

    ''' 在已有 WList 以及 bList 的前提下 rebulid model '''

    def parse_model_rebuild(self, X):
        self.echo('\nStart rebuilding model ...')

        if self.USE_MULTI:
            w_dict = self.multi_w_dict[self.net_id]
            b_dict = self.multi_b_dict[self.net_id]
            self.multi_net[self.net_id] = {}
            net = self.multi_net[self.net_id]

        else:
            w_dict = self.w_dict
            b_dict = self.b_dict
            self.net = {}
            net = self.net

        a = X

        model_len = len(self.MODEL)
        for i, config in enumerate(self.MODEL):
            _type = config['type'].lower()
            name = '%s_%d' % (_type, i + 1) if 'name' not in config else config['name']
            # self.echo('building %s layer ...' % name)

            # 卷积层
            if _type == 'conv':
                with tf.name_scope(name):
                    stride = config['stride'] if 'stride' in config else 1
                    padding = 'SAME' if 'padding' not in config or config['padding'] == 'SAME' else 'VALID'

                    a = tf.add(self.conv2d(a, w_dict[name], stride, padding), b_dict[name])

                    if 'bn' in config and config['bn']:
                        a = self.batch_normal(a, self.t_is_train, name)

                    if not 'activate' in config or config['activate']:
                        a = self.activate(a)

            # 池化层
            elif _type == 'pool':
                with tf.name_scope(name):
                    if 'pool_type' not in config or config['pool_type'] == 'max':
                        a = self.max_pool(a, config['k_size'])
                    else:
                        a = self.avg_pool(a, config['k_size'])

            # 全连接层
            elif _type == 'fc':
                with tf.name_scope(name):
                    x = tf.reshape(a, [-1, config['shape'][0]])
                    a = tf.add(tf.matmul(x, w_dict[name]), b_dict[name])

                    if ('activate' not in config and i < model_len - 1) or config['activate']:
                        a = self.activate(a)

            # 反卷积层(上采样层)
            elif _type == 'tr_conv':
                with tf.name_scope(name):
                    if 'output_shape' in config:
                        output_shape = config['output_shape']
                    elif 'output_shape_index' in config:
                        output_shape = tf.shape(net[config['output_shape_index']])
                    elif 'output_shape_x' in config:
                        output_shape = tf.shape(X)
                        for j, val_j in enumerate(config['output_shape_x']):
                            if not val_j:
                                continue
                            tmp = tf.Variable([1 if k != j else 0 for k in range(4)], tf.int8)
                            output_shape *= tmp
                            tmp = tf.Variable([0 if k != j else val_j for k in range(4)], tf.int8)
                            output_shape += tmp
                    else:
                        output_shape = None

                    stride = config['stride'] if 'stride' in config else 2
                    a = self.conv2d_transpose_stride(a, w_dict[name], b_dict[name], output_shape, stride)

            # 将上一层的输出 与 第 layer_index 层的网络相加
            elif _type == 'add':
                with tf.name_scope(name):
                    a = tf.add(a, net[config['layer_index']])

            net[name] = a

        self.echo('Finish building model')

        return a

    # **************************** 与 模型有关 的 常用函数 *************************

    ''' 激活函数 '''

    def activate(self, h):
        # return tf.multiply(h, tf.nn.sigmoid(h), name='a')
        return tf.nn.relu(h, name='a')

    ''' 2D 卷积 '''

    @staticmethod
    def conv2d(x, W, stride=1, padding='SAME'):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)

    ''' 2D 卷积 并加上 bias '''

    @staticmethod
    def conv2d_bias(x, W, b, stride=1, padding='SAME'):
        conv = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
        return tf.nn.bias_add(conv, b)

    ''' 2D 反卷积(transpose conv) 并 加上 bias '''

    @staticmethod
    def conv2d_transpose_stride(x, W, b, output_shape=None, stride=2):
        # 若没有设置 output_shape
        if isinstance(output_shape, type(None)):
            output_shape = x.get_shape().as_list()
            output_shape[1] *= stride
            output_shape[2] *= stride
            output_shape[3] = W.get_shape().as_list()[2]
        conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
        return tf.nn.bias_add(conv, b)

    ''' max pooling '''

    @staticmethod
    def max_pool(x, k_size, stride=None):
        k_size = list(hstack([1, k_size, 1]))
        if not stride:
            strides = k_size
        else:
            strides = [1, stride, stride, 1]
        return tf.nn.max_pool(x, ksize=k_size, strides=strides, padding='SAME')

    ''' mean pooling '''

    @staticmethod
    def avg_pool(x, k_size, stride=None):
        k_size = list(hstack([1, k_size, 1]))
        if not stride:
            strides = k_size
        else:
            strides = [1, stride, stride, 1]
        return tf.nn.avg_pool(x, ksize=k_size, strides=strides, padding='SAME')

    ''' batch normalize '''

    def batch_normal(self, x, is_train, name_scope):
        x_shape = x.get_shape()
        params_shape = x_shape[-1:]

        axis = list(range(len(x_shape) - 1))

        if name_scope not in self.__beta_dict:
            self.__beta_dict[name_scope] = tf.Variable(np.zeros(params_shape), name='beta', dtype=tf.float32)
        if name_scope not in self.__gamma_dict:
            self.__gamma_dict[name_scope] = tf.Variable(np.ones(params_shape), name='gamma', dtype=tf.float32)
        if name_scope not in self.__moving_mean_dict:
            self.__moving_mean_dict[name_scope] = tf.Variable(np.zeros(params_shape), name='moving_mean',
                                                              trainable=False, dtype=tf.float32)
        if name_scope not in self.__moving_std_dict:
            self.__moving_std_dict[name_scope] = tf.Variable(np.ones(params_shape), name='moving_variance',
                                                             trainable=False, dtype=tf.float32)

        mean, variance = tf.nn.moments(x, axis)

        update_moving_mean = moving_averages.assign_moving_average(self.__moving_mean_dict[name_scope],
                                                                   mean, self.BN_DECAY)
        update_moving_variance = moving_averages.assign_moving_average(self.__moving_std_dict[name_scope],
                                                                       variance, self.BN_DECAY)

        tf.add_to_collection(self.UPDATE_OPS_COLLECTION, update_moving_mean)
        tf.add_to_collection(self.UPDATE_OPS_COLLECTION, update_moving_variance)

        mean, variance = control_flow_ops.cond(is_train, lambda: (mean, variance),
                                               lambda: (self.__moving_mean_dict[name_scope],
                                                        self.__moving_std_dict[name_scope]))

        return tf.nn.batch_normalization(x, mean, variance,
                                         self.__beta_dict[name_scope], self.__gamma_dict[name_scope], self.BN_EPSILON)

    # *************************** 与 训练有关 的 常用函数 ***************************

    ''' 获取 train_op '''

    def get_train_op(self, loss, learning_rate, global_step, optimizer_func=None):
        with tf.name_scope('optimizer'):
            if isinstance(optimizer_func, type(None)):
                optimizer = tf.train.AdamOptimizer(learning_rate)
            else:
                optimizer = optimizer_func(learning_rate)

            optimizer_op = optimizer.minimize(loss, global_step=global_step)

            if not self.USE_BN:
                return optimizer_op

            batch_norm_updates = tf.get_collection(self.UPDATE_OPS_COLLECTION)
            batch_norm_updates_op = tf.group(*batch_norm_updates)
            return tf.group(optimizer_op, batch_norm_updates_op)

    # ************************* 防止 overfitting 的 trick *************************

    ''' 正则化，默认采用 l2_loss 正则化 '''

    def regularize(self, loss, beta):
        with tf.name_scope('regularize'):
            regularizer = 0.0
            w_dict = self.multi_w_dict[self.net_id] if self.USE_MULTI else self.w_dict
            for name, W in w_dict.items():
                if len(W.shape) != 2:  # 若不是全连接层的权重矩阵，则不进行正则化
                    continue
                regularizer = tf.add(regularizer, tf.nn.l2_loss(W))
            return tf.reduce_mean(loss + beta * regularizer)

    ''' 正则化，默认采用 l2_loss 正则化 '''

    def regularize_trainable(self, loss, beta):
        trainable_var = tf.trainable_variables()
        with tf.name_scope('regularize'):
            regularizer = 0.0
            for i, var in enumerate(trainable_var):
                # if i == 0:
                #     continue
                regularizer = tf.add(regularizer, tf.nn.l2_loss(tf.cast(var, tf.float32)))
        return tf.reduce_mean(loss + beta * regularizer)

    ''' dropout '''

    def dropout(self, a, keep_prob):
        return tf.nn.dropout(a, keep_prob)

    # ********************** 其他常用函数 (与 DL 无关) *********************

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

    '''
      写错误日志到 log 文件夹，会调用 traceback 将之前错误的详细信息记录到日志
    '''

    def write_log(self, err_msg=''):
        log_dir_path = os.path.join(os.path.abspath(os.path.curdir), 'log')
        if not os.path.isdir(log_dir_path):
            os.mkdir(log_dir_path)

        log_dir_path = os.path.join(log_dir_path, self.MODEL_NAME)
        if not os.path.isdir(log_dir_path):
            os.mkdir(log_dir_path)
        log_path = os.path.join(log_dir_path, 'log.txt')

        with open(log_path, 'a') as f:
            f.write("time: %s\n" % time.asctime(time.localtime(time.time())))
            if err_msg:
                if isinstance(err_msg, unicode):
                    err_msg = err_msg.encode('utf-8')
                f.write("msg: %s\n" % err_msg)
            traceback.print_exc(file=f)
            f.write("\n\n")
