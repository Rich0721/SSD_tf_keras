# SSD_tf_keras

This is a implentment SSD program. You can train myself datasets.

### Finsh 
- [x] SSD 300
- [x] VGG16 
- [x] Tensorflow lite version
- [x] SSD 512
- [ ] Mobilenet v1
- [ ] Mobilenet v2
- [ ] Mobilenet v3

```
Tensorflow verion: 2.2.0
Python version: 3.7
```

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
