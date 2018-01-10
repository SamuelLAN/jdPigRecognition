### 将每只猪识别出来

##### 根据 fcn 切割后的猪作为输入，进行识别

>#### 目录结构
- [img_arg.py](img_arg.py): 给 fcn 切割后的猪做数据增强，进行各种旋转、调光、调色等等
- [load.py](load.py): 加载数据的基类；同时也是下载数据的基类 (为了加快运行速度，同时保证不超出电脑内存限制，采用了异步加载的方式，数据在后台异步按需加载，而不是一次性全部加载到内存)
- [bi_load.py](bi_load.py): 加载数据的基类 (专门给 bi_vgg16_net.py 使用)
- [vgg16_net.py](vgg16_net.py): 使用 vgg16 模型识别猪 (图片输入大小跟 vgg 一样，为 224 * 224)
- [vgg16_net_2.py](vgg16_net_2.py): 使用 vgg16 模型识别猪 (为加快速度，图片输入大小缩小为 56 * 56)
- [vgg19_net.py](vgg19_net.py): 使用 vgg19 模型识别猪 (图片输入大小跟 vgg 一样，为 224 * 224)
- [bi_vgg16_net.py](bi_vgg16_net.py): 使用 vgg16 模型，但不是多分类，而是二分类；该程序共训练 30 个网络，每个网络进行二分类，分类目标为是该类猪与其他猪，最后将 30 个网络的训练结果根据准确率加权进行投票决定属于哪个分类 (为加快速度，图片输入大小缩小为 56 * 56)
- [resnet_50.py](resnet_50.py): 使用 resnet 50 层模型 (图片输入大小为 224 * 224)
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

> vgg16_net、vgg16_net_2 的结构图
>
> <img src="../tmp/vgg16_graph.png" alt="vgg16 的结构图" height="870" width="430">

<br>

> vgg19_net 的结构图
>
> 没保存，暂时略

<br>

> bi_vgg16_net 的结构图
>
> 没保存，暂时略

<br>

> resnet_50 的结构图
>
> 没保存，暂时略

<br>

> result
>- vgg16_net_2.py:
>> <img src="../tmp/classify_vgg16_2_cmd.png" alt="vgg16的运行结果图" height="150" width="600">
>
>- vgg16_net.py:
>> 没保存，暂时略
>
>- vgg19_net.py:
>> 没保存，暂时略
>
>- bi_vgg16_net.py:
>> 没保存，暂时略
>
>- resnet_50.py:
>> 没保存，暂时略
