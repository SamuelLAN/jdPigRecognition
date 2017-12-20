### 将每只猪识别出来

##### 根据 fcn 切割后的猪作为输入，进行识别

> 目录结构
- img_arg.py: 给 fcn 切割后的猪做数据增强，进行各种旋转、调光、调色等等
- load.py: 加载数据的基类；同时也是下载数据的基类
- vgg16_net.py: 使用 vgg16 模型识别猪
- vgg19_net.py: 使用 vgg19 模型识别猪
- get_test_csv.py: 生成 data/Test_B 对应的猪的识别结果

以上的识别模型的 loss 都使用了比赛中评分的 log_loss

> vgg16_net 的结构图

<img src="../tmp/vgg16_graph.png" alt="vgg16 的结构图" height="870" width="430">

> vgg19_net 的结构图

> bi_vgg16_net 的结构图
