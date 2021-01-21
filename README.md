# SSD_tf_keras

This is a implentment SSD program. You can train myself datasets.



### TODO 
- [x] SSD 300
- [x] VGG16 
- [x] Tensorflow lite version
- [x] SSD 512
- [x] Mobilenet v1
- [x] Mobilenet v2
- [x] Mobilenet v3
- [ ] ShuffleNet V1
- [ ] ShuffleNet V2

```
Tensorflow verion: 2.2.0
Python version: 3.7
```

## Backbone paper
I will create different CNNs use SSD. This is my reference paper.

[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)  
[MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
[Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)

## Train
Modify config.py  

CLASSES: You want to train dataset classes.  
DATASET: Train images information EX: path/001.jpg xmin,ymin,xmax,ymax,class_num. You can use `voc_annotation.py` output text file.  
```
python train.py
```

## Test
```
python predict.py
```


### Reference
[SSD: Single-Shot MultiBox Detector目标检测模型在TF2当中的实现](https://github.com/bubbliiiing/ssd-tf2/tree/67928f7e3b24a12ec0540ec09cc2b0f5406b5879)  
[SSD: Single-Shot MultiBox Detector implementation in Keras](https://github.com/mattroos/ssd_tensorflow2)  
[MobileNet V1](https://github.com/bubbliiiing/mobilenet-ssd-keras/blob/master/nets/mobilenet.py)  
[MobileNet V2](https://github.com/keras-team/keras-applications/tree/master/keras_applications)
[MobileNet V3](https://www.jianshu.com/p/9af2ae74ec04)  
