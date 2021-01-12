import xml.etree.ElementTree as ET
import os
from config import config
from glob import glob 


classes = ['1402200300101', '1402300300101', '1402310200101', '1402312700101', '1402312900101', 
        '1402324800101', '1422001900101', '1422111300101', '1422204600101', '1422206800101', '1422300300101', 
        '1422301800101', '1422302000101', '1422308000101', '1422329600101', '1422503600101', '1422504400101', 
        '1422505200101', '1422505600101', '1422593400101', '1422594600101']


datasets = "../datasets/test"

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

if __name__ == "__main__":
    
    images = glob(os.path.join(config.DATASET, "JPEGImages", "*.jpg"))
    xmls = glob(os.path.join(config.DATASET, "Annotations", "*.xml"))
    assert len(images) == len(xmls), "Images number and XML files number need same."

    for image, xml in zip(images, xmls):
        print(image)