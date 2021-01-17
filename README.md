# 人脸识别-重识别技术

## 一、环境说明

- python  3.7.6
- torch  1.3.1
- torchvision  0.4.2
- pillow  7.1.2
- numpy  1.19.2
- scipy  1.2.2

------

## 二、目录结构

- database
  - 存放图片
- features
  - 存放提取的图像特征
- model
  - ResNet50.py  模型
- query
  - 存放要识别的人脸图
- utils
  - get_name.py  转换名字
  - similarity_measurement.py  相似度指标
- query.py
- update_the_database.py

## 三、使用方法

```
$ python update_the_database.py  ## 用来更新数据库
$ python query.py  ## 识别
```

