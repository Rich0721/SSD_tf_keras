
class config:

    EPOCHS = 50
    BATCH_SIZE = 8
    IMAGE_SIZE_300 = (300, 300, 3)
    IMAGE_SIZE_512 = (512, 512, 3)
    ANCHORS_SIZE_300 = [30, 60, 111, 162, 213, 264, 315] # VOC SSD300
    #ANCHORS_SIZE_300 = [21, 45, 99, 153, 207, 261, 315] # COCO
    ANCHORS_SIZE_512 = [36, 77, 154, 230, 307, 384, 461, 538] # VOC SSD512
    #ANCHORS_SIZE_512 = [20, 51, 133, 215, 297, 379, 461, 542] # COCO
    VARIANCES = [0.1, 0.1, 0.2, 0.2]

    '''
    # VOC dataset classes   
    CLASSES =  ['background', 'airwaves-mint', 'eclipse-lemon', 'eclipse-mint', 'eclipse-mint-fudge',
           'extra-lemon', 'hallsxs-buleberry', 'hallsxs-lemon', 'meiji-blackchocolate', 'meiji-milkchocolate', 'rocher']
    '''
    CLASSES = ['1402200300101', '1402300300101', '1402310200101', '1402312700101', '1402312900101', 
        '1402324800101', '1422001900101', '1422111300101', '1422204600101', '1422206800101', '1422300300101', 
        '1422301800101', '1422302000101', '1422308000101', '1422329600101', '1422503600101', '1422504400101', 
        '1422505200101', '1422505600101', '1422593400101', '1422594600101']
    '''
    # COCO dataset classes
    CLASSES = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 
        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 
        'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
         'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 
         'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 
         'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    '''
    
    DATASET = "../datasets/test"
    TRAIN_TEXT = "./train.txt"
    TEST_DATASET  = "../datasets/test_network"

    COCO_JSON = ['instances_train2014.json', 'instances_val2014.json']
    COCO_DATASER_FOLDER = "../datasets/coco_train2014/"
    COCO_TRAIN_TEXT = ["./train.txt", "./val.txt"]

    CONFIDENCE = 0.5
    NMS_IOU = 0.45
    MODEL_FOLDER = "./logs_2/"
    FILE_NAME = "ssd_vgg"