import xml.etree.ElementTree as ET
import os

classes = ['airwaves-mint', 'eclipse-lemon', 'eclipse-mint', 'eclipse-mint-fudge',
           'extra-lemon', 'hallsxs-buleberry', 'hallsxs-lemon', 'meiji-blackchocolate',
           'meiji-milkchocolate', 'rocher']

datasets = "../datasets/300"

def convert_annotation(xml_file, text_file):

    xml_file = os.path.join(datasets, "Annotations", xml_file + ".xml")
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for obj in root.iter("object"):
        difficult = obj.find("difficult").text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        bndbox = obj.find('bndbox')
        bbox = (int(bndbox.find("xmin").text), int(bndbox.find("ymin").text), int(bndbox.find("xmax").text), int(bndbox.find("ymax").text))
        text_file.write(" " + ",".join([str(b) for b in bbox]) + "," + str(cls_id))

image_ids = open("../preprocess/train_300.txt").read().strip().split()
train_text = open("train.txt", "w")
for image_id in image_ids:
    
    image_path = os.path.join(datasets, "JPEGImages", image_id + ".jpg")
    print("Image path:{}".format(image_path))
    train_text.write(image_path)
    convert_annotation(xml_file=image_id, text_file=train_text)
    train_text.write("\n")
train_text.close()

