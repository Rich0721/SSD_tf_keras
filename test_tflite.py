import tensorflow as tf
import numpy as np
import cv2
import pathlib
import os
import xml.etree.ElementTree as ET
from keras.preprocessing import image
from utils.utils import BBoxUtility, letterbox_image, ssd_correct_boxes
from PIL import Image, ImageFont, ImageDraw
model_path = "logsSSD512_VGG16-100.tflite"

img_height = 300
img_width = 300

datasets = "../datasets/300"
test = '../preprocess/test_300.txt'

classes = [
           'airwaves-mint', 'eclipse-lemon', 'eclipse-mint', 'eclipse-mint-fudge',
           'extra-lemon', 'hallsxs-buleberry', 'hallsxs-lemon', 'meiji-blackchocolate',
           'meiji-milkchocolate', 'rocher']
bbox_util = BBoxUtility(num_classes=len(classes) + 1)

with open(test) as f:
    test_lines = f.readlines()
np.random.seed(1000)
np.random.shuffle(test_lines)
images = []
annotations = []

for line in test_lines:
    image_path = os.path.join(datasets, "JPEGImages", line[:-1] + ".jpg")
    annotation_path = os.path.join(datasets, "Annotations", line[:-1] + ".xml")

    images.append(image_path)
    annotations.append(annotation_path)

def read_xml(xml_file):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    obj = root.find("object")
    name = obj.find("name").text
    bndbox = obj.find("bndbox")
    xmin = int(bndbox.find("xmin").text)
    ymin = int(bndbox.find("ymin").text)
    xmax = int(bndbox.find("xmax").text)
    ymax = int(bndbox.find("ymax").text)

    return [name, xmin, ymin, xmax, ymax]


interpreter = tf.lite.Interpreter(model_path=model_path)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
print(input_details)
output_details = interpreter.get_output_details()
print(output_details)

def detect_image(image):

    image_shape = np.array(np.shape(image)[0:2])
    crop_img, x_offset, y_offset = letterbox_image(image, (300, 300))
    photo = np.array(crop_img, dtype=np.float32)
    photo = np.expand_dims(photo, axis=0)
    interpreter.set_tensor(input_details[0]['index'], photo)
    
    interpreter.invoke()
    
    rects = interpreter.get_tensor(output_details[0]['index'])
    
    results = bbox_util.detection_out(predictions=rects, conf_threshold=0.5)
    
    if len(results[0]) <= 0:
        return image

    det_labels = results[0][:, 0]
    det_confs = results[0][:, 1]
    det_xmin, det_ymin, det_xmax, det_ymax = results[0][:, 2], results[0][:, 3], results[0][:, 4], results[0][:, 5]
    top_indices = [i for i, conf in enumerate(det_confs) if conf >= 0.5]
    top_conf = det_confs[top_indices]
    top_label_indices = det_labels[top_indices].tolist()
    top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(det_xmin[top_indices], -1), np.expand_dims(det_ymin[top_indices], -1), np.expand_dims(det_xmax[top_indices], -1), np.expand_dims(det_ymax[top_indices], -1)
    
    boxes = ssd_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax, np.array([300, 300]), image_shape)
    print(boxes)
    font = ImageFont.truetype(font='simhei.ttf',size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
    thickness = (np.shape(image)[0] + np.shape(image)[1]) // 300
    
    for i, c in enumerate(top_label_indices):
        predicted_class = classes[int(c)-1]
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
                outline=(0, 255, 255))
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=(0, 255, 255))
        draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
        del draw
    return image

img = "../datasets/300/JPEGImages/1112.jpg"
image = Image.open(img)
r_image = detect_image(image)
r_image.show()


    
    
    







    