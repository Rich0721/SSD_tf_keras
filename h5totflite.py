#from keras import models
import tensorflow as tf
#from tensorflow.keras.models import load_model
from tensorflow.python.keras import backend as K
from models.ssd_300 import SSD300
from config import config


weight_path = "logs/ssd_vgg_epoch48.h5"
tf_weight_path = "logs/ssd_vgg_epoch48.tflite"

img_height = 300
img_width = 300


model = SSD300(config.IMAGE_SIZE, len(config.CLASSES))
model.load_weights(weight_path, by_name=True)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.allow_custom_ops = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

tfmodel = converter.convert()
open(tf_weight_path, 'wb').write(tfmodel)
