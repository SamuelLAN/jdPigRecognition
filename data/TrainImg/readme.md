## 存放训练集数据的文件夹

#### 其中包含：

- video_process 生成的图片 (没做 data argument)

- fcn 对 video_process 切割后的猪的图片

### 使用对象：

- fcn 会在这里生成猪的切割的图片

- classify 做 data argument 时需要用到这里 fcn 生成的猪的切割图片
