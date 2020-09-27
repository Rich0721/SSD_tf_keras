#from keras import models
import tensorflow as tf
#from tensorflow.keras.models import load_model
from tensorflow.python.keras import backend as K
from nets.ssd_vgg import SSD300_VGG16


weight_path = "logsSSD512_VGG16-100.h5"
tf_weight_path = "logsSSD512_VGG16-100.tflite"

img_height = 300
img_width = 300


model = SSD300_VGG16((300, 300, 3), 11)
model.load_weights(weight_path, by_name=True)
sess = K.get_session()
#converter = tf.lite.TFLiteConverter.from_keras_model_file(weight_path ,custom_objects=custom_objects)
converter =tf.lite.TFLiteConverter.from_session(sess, model.inputs, model.outputs)
converter.allow_custom_ops = True
#converter.target_ops = [tf.lite.OpsSet.SELECT_TF_OPS, tf.lite.OpsSet.TFLITE_BUILTINS]

tflite_model = converter.convert()
open(tf_weight_path, "wb").write(tflite_model)