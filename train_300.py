<<<<<<< Updated upstream:train_300.py
from __future__ import annotations
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from ssd_keras_layers.ModelCheckpoint import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from models.ssd_300 import SSD300
from loss.ssd_loss import ssd_loss
from generator import Generator
from utils.anchors import get_anchors_300
from utils.utils import BBoxUtility
from config import config

################init set################################
img_height = 300 # Height of the model input images
img_width = 300 # Width of the model input images
img_channels = 3 # Number of color channels of the model input images
anchors = [30, 60, 111, 162, 213, 264, 315]
variances = [0.1, 0.1, 0.2, 0.2]

epochs = 50
batch_size = 8

text_file = "./train.txt" # train images information text file.
model_save_folder = "./logs/"
h5_file_name = "ssd_vgg_300"
########################################################

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if not os.path.exists(model_save_folder):
    os.mkdir(model_save_folder)

if __name__ == "__main__":

    priors = get_anchors_300((img_height, img_width))
    
    bbox_util = BBoxUtility(len(config.CLASSES), priors)

    model = SSD300((img_height, img_width, img_channels),
                    n_classes=len(config.CLASSES),
                    anchors=anchors,
                    variances=variances)

    checkpoint = ModelCheckpoint(model_save_folder + "ssd_vgg_epoch{epoch:02d}.h5",
                monitor='val_loss', save_weights_only=True, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    
    val_split = 0.1
    with open(text_file) as f:
        lines = f.readlines()
    np.random.seed(1000)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    gen = Generator(bbox_util, batch_size, lines[:num_train], lines[num_train:], (img_height, img_width), len(config.CLASSES))

    model.compile(optimizer=Adam(lr=5e-4), loss=ssd_loss(len(config.CLASSES)).compute_loss)
    model.fit_generator(gen.generator(True),
                        steps_per_epoch=num_train//batch_size,
                        validation_data=gen.generator(False),
                        validation_steps=num_val//batch_size,
                        epochs=epochs,
                        initial_epoch=0,
                        callbacks=[checkpoint, reduce_lr])
=======
from __future__ import annotations
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from ssd_keras_layers.ModelCheckpoint import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from models.ssd_300 import SSD300
from loss.ssd_loss import ssd_loss
from generator import Generator
from utils.anchors import get_anchors_300
from utils.utils import BBoxUtility
from config import config

################init set################################
img_height = 300 # Height of the model input images
img_width = 300 # Width of the model input images
img_channels = 3 # Number of color channels of the model input images
anchors = [30, 60, 111, 162, 213, 264, 315]
variances = [0.1, 0.1, 0.2, 0.2]

epochs = 50
batch_size = 8

text_file = "./train.txt" # train images information text file.
model_save_folder = "./logs/"
h5_file_name = "ssd_vgg_300"
########################################################

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if not os.path.exists(model_save_folder):
    os.mkdir(model_save_folder)

if __name__ == "__main__":

    priors = get_anchors_300((img_height, img_width))
    
    bbox_util = BBoxUtility(len(config.CLASSES), priors)

    model = SSD300((img_height, img_width, img_channels),
                    n_classes=len(config.CLASSES),
                    anchors=anchors,
                    variances=variances)

    checkpoint = ModelCheckpoint(model_save_folder + "ssd_vgg_epoch{epoch:02d}.h5",
                monitor='val_loss', save_weights_only=True, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    
    val_split = 0.1
    with open(text_file) as f:
        lines = f.readlines()
    np.random.seed(1000)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    gen = Generator(bbox_util, batch_size, lines[:num_train], lines[num_train:], (img_height, img_width), len(config.CLASSES))

    model.compile(optimizer=Adam(lr=5e-4), loss=ssd_loss(len(config.CLASSES)).compute_loss)
    model.fit_generator(gen.generator(True),
                        steps_per_epoch=num_train//batch_size,
                        validation_data=gen.generator(False),
                        validation_steps=num_val//batch_size,
                        epochs=epochs,
                        initial_epoch=0,
                        callbacks=[checkpoint, reduce_lr])
>>>>>>> Stashed changes:train.py
    model.save(model_save_folder, save_format='tf')