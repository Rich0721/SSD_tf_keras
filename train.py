from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler, TerminateOnNaN, CSVLogger
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing import image
from nets.ssd_vgg import SSD300_VGG16
from nets.ssd_training import MultiboxLoss, Generator
from utils.utils import BBoxUtility
from utils.anchors import get_anchors
from tensorflow.python.keras.optimizers import Adam
import numpy as np
import tensorflow as tf
import cv2
import os
import sys

log_dir = "logs"
train_annotation_path = "train.txt"
val_annotation_path = "val.txt"
batch_size = 4
NUM_CLASSES = 11
input_shape = (300, 300, 3)

file_name = "SSD512_VGG16"



def lr_schedule(epoch):
    if epoch < 30:
        return 5e-4
    elif epoch < 50:
        return 2e-4
    else:
        return 1e-4

if __name__ == "__main__":

    priors = get_anchors((input_shape[0], input_shape[1]))
    bbox_util = BBoxUtility(NUM_CLASSES, priors)

    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    

    np.random.seed(1000)
    np.random.shuffle(train_lines)
    np.random.shuffle(val_lines)
    np.random.seed(None)
    num_train = len(train_lines)
    num_val = len(val_lines)

    model = SSD300_VGG16(input_shape, num_classes=NUM_CLASSES)

    # 參數設定
    logging = TensorBoard(log_dir=log_dir)
    csv_logger = CSVLogger(filename=os.path.join(log_dir, file_name + "_logger.csv"))
    checkpoint = ModelCheckpoint(os.path.join(log_dir, file_name + "-{epoch:03d}.h5"))
    learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule, verbose=1)
    terminate_on_nan = TerminateOnNaN()
    callbacks = [logging, csv_logger, checkpoint, learning_rate_scheduler, terminate_on_nan]

    # GPU 使用量設定
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.set_session(sess)

    gen = Generator(bbox_util, batch_size, train_lines, val_lines, (input_shape[0], input_shape[1]), NUM_CLASSES)

    model.compile(optimizer=Adam(lr=5e-4), loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=3.0).compute_loss)
    model.fit_generator(gen.generate(True),
                        steps_per_epoch=num_train // batch_size,
                        validation_data=gen.generate(False),
                        validation_steps=num_val // batch_size,
                        epochs=100, initial_epoch=0,
                        callbacks=callbacks)
    
