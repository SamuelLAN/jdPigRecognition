### 将每只猪识别出来

##### 根据 fcn 切割后的猪作为输入，进行识别

>#### 目录结构
- [img_arg.py](img_arg.py): 给 fcn 切割后的猪做数据增强，进行各种旋转、调光、调色等等
- [load.py](load.py): 加载数据的基类；同时也是下载数据的基类 (为了加快运行速度，同时保证不超出电脑内存限制，采用了异步加载的方式，数据在后台异步按需加载，而不是一次性全部加载到内存)
- [bi_load.py](bi_load.py): 加载数据的基类 (专门给 [bi_vgg16_net.py](bi_vgg16_net.py) 使用)
- [vgg16_net.py](vgg16_net.py): 使用 vgg16 模型识别猪 (图片输入大小跟 vgg 一样，为 224 * 224)
- [vgg16_net_2.py](vgg16_net_2.py): 使用 vgg16 模型识别猪 (为加快速度，图片输入大小缩小为 56 * 56)
- [vgg19_net.py](vgg19_net.py): 使用 vgg19 模型识别猪 (为加快速度，图片输入大小缩小为 56 * 56)
- [bi_vgg16_net.py](bi_vgg16_net.py): 使用 vgg16 模型，但不是多分类，而是二分类；该程序共训练 30 个网络，每个网络进行二分类，分类目标为是该类猪与其他猪，最后将 30 个网络的训练结果根据准确率加权进行投票决定属于哪个分类 (为加快速度，图片输入大小缩小为 56 * 56)
- [resnet_50.py](resnet_50.py): 使用 resnet 50 层模型 (图片输入大小为 224 * 224); resnet 还没试过运行，之后有时间会尝试运行
- [get_test_csv.py](get_test_csv.py): 生成 data/Test_B 对应的猪的识别结果

<br>

>#### 训练方法
> 由于比赛评分标准为 log_loss (具体自己参考比赛官网的公式)，因此训练模型时，使用了普通的 loss 与 log_loss 交互训练的方法
>
>> 1、先使用普通的 loss 进行训练，调整参数，训练过程将校验集准确率最高的结果保存下来，直到停止或 early_stop 为止；
>>
>> 2、在 1 训练好的模型的基础上，将 loss function 改成 log_loss 进行训练，调整参数，训练过程将检验集 log_loss 最低的结果保存下来，直到停止或 early_stop 为止；
>>
>> 3、重复 1 与 2 的过程，直到达到理想效果

<br>

>### vgg16_net、vgg16_net_2
> vgg16_net 与 vgg16_net_2 的区别在与输入大小的不同，导致网络全连接层参数数量不一样
>
> vgg16_net_2 比 vgg16_net 运行速度更快，因为参数较少
>
> 此处的 vgg 模型，加入了 batch_normalize，为了加快训练速度
>
>##### 结构图
>
> <img src="../tmp/vgg16_graph.png" alt="vgg16 的结构图" height="1260" width="600">
>
>##### vgg16_net_2 的运行结果
>
> <img src="../tmp/classify_vgg16_2_cmd.png" alt="vgg16的运行结果图" height="150" width="660">
>
> 由于 vgg16_net 与 vgg16_net_2 运行效果差不多，而 vgg16_net 运行较慢，所以没有等 vgg16_net 运行完，这里暂不提供运行结果

<br>

>#### bi_vgg16_net
> 该方法将 30 分类 转化为 30 个 二分类问题，需要训练 30 个网络，最后根据各个网络的准确率作为权重，加权投票作为输出
>
> <strong>优点</strong>：可扩展性，当新增分类时，无需重新训练全部数据，只需针对新分类训练一个新的网络即可
>
> <strong>致命缺点</strong>：尽管单个网络的准确率能高达 90+%，但当需要整合 30 个网络时，能保证不出错的概率就变成 0.9 ^ 30 = 0.0424 ，这是一个非常小的数字，意味着当综合考虑时，总会有一些网络会出错出现干扰，导致准确率无法提升
>
> 由于该模型存在致命缺点，这里就不展示它的结构图了，准确率只有 60% 多
>
> 其中 30 个网络的准确率以及 log_loss
>
> <img src="../tmp/classify_bi_vgg_result_1.png" alt="bi_vgg16的运行结果图" height="700" width="660">
>
> <img src="../tmp/classify_bi_vgg_result_2.png" alt="bi_vgg16的运行结果图" height="700" width="660">
>
> <img src="../tmp/classify_bi_vgg_result_3.png" alt="bi_vgg16的运行结果图" height="700" width="660">
>
> 可见单个网络的准确率可以很高，最高 97% - 98%，若给每个网络调一下参数，平均准确率应该能高于 90%；但是关键就在于上面所说的致命弱点
>
> 这里就不展示最终合并在一起的准确率的运行结果图了，之前忘记截图，准确率就 60+%

<br>

> vgg19_net 的结构图
>
> 此处的 vgg 模型，加入了 batch_normalize，为了加快训练速度
>
> <img src="../tmp/vgg19_graph.png" alt="vgg16 的结构图" height="1300" width="600">
>
> tensorboard 的截图
>
> <img src="../tmp/classify_vgg19_tensorboard.png" alt="bi_vgg16的运行结果图" height="540" width="460">
>
> 还没仔细地调参数，目前准确率就 60+%，跟 bi_vgg16 的效果差不多，运行结果图忘记截图了

<br>

> resnet_50 的结构图
>
> 没保存，暂时略
