from unittest import result
import cv2
import numpy as np
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input
from tensorflow.python.keras.backend import dtype
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import backend as K
from nets.ssd_vgg import SSD300_VGG16
from PIL import Image, ImageFont, ImageDraw
import colorsys, os
import os
from utils.utils import BBoxUtility, letterbox_image, ssd_correct_boxes

class test_ssd:

    def __init__(self, weight_file, classes):
        
        self.classes = classes
        self.sess = K.get_session()
        self.image_size = (300, 300, 3)
        self.confidence = 0.5
        self.generate(weight_file=weight_file)
        self.bbox_util = BBoxUtility(self.num_classes)

    def generate(self, weight_file):

        self.num_classes = len(self.classes) + 1

        self.model = SSD300_VGG16(self.image_size, self.num_classes)
        self.model.load_weights(weight_file, by_name=True)

        # set all classes colors
        hsv_tuples = [(x / len(self.classes), 1, 1.) for x in range(len(self.classes))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0]*255), int(x[1]*255), int(x[2]*255)), self.colors))

    def detect_image(self, image):

        image_shape = np.array(np.shape(image)[0:2])
        crop_img, x_offset, y_offset = letterbox_image(image, (self.image_size[0], self.image_size[1]))
        photo = np.array(crop_img, dtype=np.float64)

        photo = preprocess_input(np.reshape(photo, [1, self.image_size[0], self.image_size[1], 3]))
        predicts = self.model.predict(photo)
        
        results = self.bbox_util.detection_out(predictions=predicts, conf_threshold=self.confidence)
        
        if len(results[0]) <= 0:
            return image

        det_labels = results[0][:, 0]
        det_confs = results[0][:, 1]
        det_xmin, det_ymin, det_xmax, det_ymax = results[0][:, 2], results[0][:, 3], results[0][:, 4], results[0][:, 5]
        top_indices = [i for i, conf in enumerate(det_confs) if conf >= self.confidence]
        top_conf = det_confs[top_indices]
        top_label_indices = det_labels[top_indices].tolist()
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(det_xmin[top_indices], -1), np.expand_dims(det_ymin[top_indices], -1), np.expand_dims(det_xmax[top_indices], -1), np.expand_dims(det_ymax[top_indices], -1)
        
        boxes = ssd_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax, np.array([self.image_size[0], self.image_size[1]]), image_shape)
        font = ImageFont.truetype(font='simhei.ttf',size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = (np.shape(image)[0] + np.shape(image)[1]) // self.image_size[0]
        
        for i, c in enumerate(top_label_indices):
            predicted_class = self.classes[int(c)-1]
            score = top_conf[i]
            
            ymin, xmin, ymax, xmax = boxes[i]

            xmin = max(0, int(xmin))
            ymin = max(0, int(ymin))
            xmax = min(np.shape(image)[0], int(xmax))
            ymax = min(np.shape(image)[1], int(ymax))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')

            if ymin - label_size[1] >= 0:
                text_origin = np.array([xmin, ymin - label_size[1]])
            else:
                text_origin = np.array([xmin, ymin + 1])
            
            for i in range(thickness):
                draw.rectangle(
                    [xmin + i, ymin + i, xmax - i, ymax - i],
                    outline=self.colors[int(c)-1])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[int(c)-1])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image
    
    def close_session(self):
        self.sess.close()

if __name__ == "__main__":
    
    weight_path = "./logsSSD512_VGG16-100.h5"
    classes = ['airwaves-mint', 'eclipse-lemon', 'eclipse-mint', 'eclipse-mint-fudge',
                'extra-lemon', 'hallsxs-buleberry', 'hallsxs-lemon', 'meiji-blackchocolate',
                'meiji-milkchocolate', 'rocher']
    predict = test_ssd(weight_path, classes=classes)

    img = "../datasets/300/JPEGImages/1112.jpg"
    image = Image.open(img)
    r_image = predict.detect_image(image)
    r_image.show()

    predict.close_session()
