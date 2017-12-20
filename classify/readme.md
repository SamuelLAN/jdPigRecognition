### 将每只猪识别出来

##### 根据 fcn 切割后的猪作为输入，进行识别

> 目录结构
- img_arg.py: 给 fcn 切割后的猪做数据增强，进行各种旋转、调光、调色等等
- load.py: 加载数据的基类；同时也是下载数据的基类
- vgg16_net.py: 使用 vgg16 模型

> vgg16_net 结构图

<img src="../tmp/vgg16_graph.png" alt="vgg16 的结构图" height="870" width="430">

