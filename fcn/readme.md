# FCN 全卷积神经网络

##### 负责将猪从图片中切割出来，也就是去掉背景，只留下猪

> FCN 的结构图

<img src="../tmp/fcn_cmd.png" alt="FCN 的结构图" height="100" width="220">

> FCN 训练过程 tensorboard 的 scalar

<img src="../tmp/fcn_scalar.png" alt="FCN 训练过程 tensorboard 的 scalar" height="210" width="420">

> FCN 训练过程 tensorboard 里的 image
>> 其中 input_image 为输入图像；output_image 为切割后的图像；truth image 为 ground truth

<img src="../tmp/fcn_img_1.png" alt="FCN 训练过程 tensorboard tensorboard 里的 image 1" height="381" width="431">
<img src="../tmp/fcn_img_2.png" alt="FCN 训练过程 tensorboard tensorboard 里的 image 2" height="381" width="431">
<img src="../tmp/fcn_img_3.png" alt="FCN 训练过程 tensorboard tensorboard 里的 image 3" height="381" width="431">

> FCN 运行结果的 cmd 图

<img src="../tmp/fcn_cmd.png" alt="FCN 运行结果的 cmd 图" height="100" width="210">

> 文档结构
- load.py: 加载数据的基类
- fcn.py: fcn 的模型；继承于 lib/base，运行里面的 FCN.run 即可训练模型
- get_image.py: 引用 fcn.py 将 data/TrainImg 里的猪切割出来
- get_test_image.py: 引用 fcn.py 将 data/Test_B 里的猪切割出来

