from six import assertRaisesRegex
from config import config
from ssd_predict import detector
from PIL import Image
from glob import glob
import os
from config import config
from datetime import datetime
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != "GPU"


det = detector(weight_path="./logs/ssd_vgg_epoch48.h5")
images = glob(os.path.join(config.TEST_DATASET, "JPEGImages", "*.jpg"))
#image = "../datasets/test/JPEGImages/09028.jpg"
start = datetime.now()
for image in images:
    print("Image file:{}".format(image))
    img = Image.open(image)
    r_image = det.detect_image(img)
    #r_image.show()
end = datetime.now()
print("{}".format(end-start))