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
>
> 2、tensorflow 1.0+、numpy、six

<br>

>##### 注意事项
> 1、之前收到反馈，运行 fcn 时报内存不足问题，这里稍作解释；由于本人自己的电脑内存还算足够，因此 fcn 的数据加载方式采用的是一次性全部加载的内存的方式；若电脑内存不是很大的比如只有 8 G，可以适当修改 fcn 里面的 load.py，改成按需加载，建议可以参考 classify/load.py 里的数据加载方式，里面采用的是异步按需加载，既保证了内存问题，也保证了速度

<br>

>##### 比赛排名
> 虽然这次比赛排名不理想，总共1386支队伍参加，B榜排名101 ／ 1386 名；<br>
但以 classify 得到的结果，校验集的 <strong style="color: red; font-weight: normal;">log_loss</strong> 只有 <strong style="color: red; font-weight: normal;">0.32</strong>，是在前 20 名并能进决赛的；<br>可惜之前 save 模型没做好，导致 restore 的时候出错，没法 restore 出训练的结果；<br>当发现该问题时再重新跑程序已经不够时间提交了，以致排名只有 101 名
>

<br>

>#### 感想
> 这次参加比赛比较晚，参加比赛时已经过了一半时间，加上自己前面分配时间不太妥当，悠哉悠哉地进行，导致后期虽然完成了，但却不够时间跑程序以及 debug 问题；
>
> 总的来说，发现了自己的几点问题
>
> 1、看论文不够多，思路不够宽
>
> 2、分配时间不够妥当; 把大部分时间浪费在把猪抠出来上，而没有放在真正重要的分类上；
>
> 一开始老想着不手动标注，想用无监督的方法把猪抠出来，但尝试了很多都没有效果，白白浪费时间；后来再转 fcn 时，已经浪费很多时间了
