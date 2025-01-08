## 环境

代码编写环境为Python3.8，Pytorch为CUDA12.4对应的最新版torch

## 准备数据

下载MSCOCO2014数据集中的train和val并放置在./data/images/directory中，格式如下

```bash
./data/images/directory/
|- train2014/
|-|- COCO_train2014_000000000009.jpg
|-|- ......

|- val2014/
|-|- COCO_val2014_000000000042.jpg
|-|- ......
```

图像需要被调整大小，运行代码resize_images.py

```bash
python resize_images.py
```

下载annotations，放置在data目录中，然后运行

```python
python KarpathySplit.py
```

下载图像概念，链接为https://drive.google.com/open?id=1jpSZbLXD1Ev3OC2t_NFFvxYo40UcnV7Q

放置在data目录中

## 训练

* Baseline

```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py --basic_model=VisualAttention 
```

* MIA

```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py --basic_model=VisualAttention --use_MIA=True --iteration_times=2
```

* PTP

```python
# 在Train.py文件中找到第11行 把
from data_loader import get_loader 
# 修改为
from ptp_data_loader import get_loader
# 即可使用PTP预处理完成的数据
```

## 预测

* Baseline

```bash
CUDA_VISIBLE_DEVICES=0 python test.py  --basic_model=VisualAttention 
```

* MIA

```bash
CUDA_VISIBLE_DEVICES=0 python test.py  --basic_model=VisualAttention  --use_MIA=True --iteration_times=2
```

## 关于checkpoint

本次实验中的四个结果checkpoint文件被放在链接https://pan-yz.cldisk.com/external/m/file/1076174840539475968

## 关于评价结果

本次实验的评价结果均存储在./evaluation文件夹中
