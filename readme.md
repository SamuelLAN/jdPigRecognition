# 京东猪脸识别比赛

> link: http://jddjr.jd.com/item/4

### 该项目共分 3 个流程

- video_process

    将视频处理成图片数据

    data argument, 对所有图片进行旋转、平移、调亮度、调色度...等等，丰富训练集

    ( 具体看文件夹 [video_process](video_process) ，里面附有 [readme](video_process/readme.md) )

- fcn

    处理图片数据，将猪单独切割处理

    data argument，将切割后的猪的图片进行旋转、平移、调亮度、调色度...等等，丰富训练集

    ( 具体看文件夹 [fcn](fcn) ，里面附有 [readme](fcn/readme.md) )

- classify

    将每只猪识别出来

    根据 fcn 切割后的猪作为输入，进行识别

    ( 具体看文件夹 [classify](classify) ，里面附有 [readme](classify/readme.md) )

<br>

>##### 环境
> 1、python 环境，该代码可兼容 python 2.7、3.5、3.6
> 2、tensorflow 1.0+、numpy、six

<br>

>##### 比赛排名
> 虽然这次比赛排名不理想，总共1386支队伍参加，B榜排名101名；<br>
但以 classify 得到的结果，校验集的 log_loss 只有 0.53，是能排在前 30 名的；<br>可惜之前 save 模型没做好，导致 restore 的时候出错，没法 restore 出训练的结果；<br>当发现该问题时再重新跑程序已经不够时间提交了，以致排名只有 101 名

<br>

>##### 注意事项
> 1、之前收到反馈，运行 fcn 时报内存不足问题，这里稍作解释；由于本人自己的电脑内存还算足够，因此 fcn 的数据加载方式采用的是一次性全部加载的内存的方式；若电脑内存不是很大的比如只有 8 G，可以适当修改 fcn 里面的 load.py，改成按需加载，建议可以参考 classify/load.py 里的数据加载方式，里面采用的是异步按需加载，既保证了内存问题，也保证了速度
