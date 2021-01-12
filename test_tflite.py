import tensorflow as tf
import numpy as np
import cv2
import pathlib
import os
import xml.etree.ElementTree as ET
from tensorflow.keras.preprocessing import image
from utils.utils import BBoxUtility,letterbox_image,ssd_correct_boxes
from PIL import Image,ImageFont, ImageDraw
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import datetime
model_path = "./logs/ssd_vgg_epoch48.tflite"

img_height = 300
img_width = 300

datasets = "../datasets/test_network"
test = '../datasets/test_network/train.txt'

classes = ['1402200300101', '1402300300101', '1402310200101', '1402312700101', '1402312900101', 
        '1402324800101', '1422001900101', '1422111300101', '1422204600101', '1422206800101', '1422300300101', 
        '1422301800101', '1422302000101', '1422308000101', '1422329600101', '1422503600101', '1422504400101', 
        '1422505200101', '1422505600101', '1422593400101', '1422594600101']
bbox_util = BBoxUtility(len(classes)-1)

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

def predict(img):
    img = np.array(img, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], img)
    
    interpreter.invoke()

    rects = interpreter.get_tensor(output_details[0]['index'])
    return rects

def detect_image(image ,model_image_size=(300, 300, 3)):
        
        
        image_shape = np.array(np.shape(image)[0:2])
        crop_img,x_offset,y_offset = letterbox_image(image, (model_image_size[0],model_image_size[1]))
        photo = np.array(crop_img,dtype = np.float64)

        # 图片预处理，归一化
        photo = preprocess_input(np.reshape(photo,[1,model_image_size[0],model_image_size[1],3]))
        preds = predict(photo)

        # 将预测结果进行解码
        results = bbox_util.detection_out(preds, confidence_threshold=0.5)
        
        if len(results[0])<=0:
            return image

        # 筛选出其中得分高于confidence的框
        det_label = results[0][:, 0]
        det_conf = results[0][:, 1]
        det_xmin, det_ymin, det_xmax, det_ymax = results[0][:, 2], results[0][:, 3], results[0][:, 4], results[0][:, 5]
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.5]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(det_xmin[top_indices],-1),np.expand_dims(det_ymin[top_indices],-1),np.expand_dims(det_xmax[top_indices],-1),np.expand_dims(det_ymax[top_indices],-1)
        
        # 去掉灰条
        boxes = ssd_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([model_image_size[0],model_image_size[1]]),image_shape)
        
        
        font = ImageFont.truetype(font='simhei.ttf',size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = (np.shape(image)[0] + np.shape(image)[1]) // model_image_size[0]

        for i, c in enumerate(top_label_indices):
            predicted_class = classes[int(c)-1]
            score = top_conf[i]

            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label)
            label = label.encode('utf-8')
            print(label)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=(255, 0, 255))
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=(255, 255, 255))
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        
        return image



interpreter = tf.lite.Interpreter(model_path=model_path)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
#print(input_details)
output_details = interpreter.get_output_details()
#print(output_details)

start_time = datetime.datetime.now()
for image_path in images:
    
    #name, xmin, ymin, xmax, ymax = read_xml(xml)
    
    img = Image.open(image_path)
    img = detect_image(img)


    #img.show()
    
    print("Image file:{}".format(image_path))
end_time = datetime.datetime.now()
print("{}".format(end_time-start_time))