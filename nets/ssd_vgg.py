from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Activation, Conv2D, Dense, Flatten
from tensorflow.python.keras.layers import GlobalAveragePooling2D, MaxPooling2D, Input
from tensorflow.python.keras.layers import merge, concatenate
from tensorflow.python.keras.layers import Reshape, ZeroPadding2D
from tensorflow.python.keras.models import Model
from nets.ssd_layers import Normalize, PriorBox

def SSD300_VGG16(input_shape, num_classes=21):

    input_tensor = Input(shape=input_shape)
    img_size = (input_shape[1], input_shape[0])

    ####### VGG 16 architecture ##########################
    # Block 1
    conv1_1   = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(input_tensor)
    conv1_2  = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(conv1_1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1')(conv1_2)

    # Block 2
    conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(pool1)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(conv2_1)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2')(conv2_2)

    # Block 3
    conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(pool2)
    conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(conv3_1)
    conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(conv3_2)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3')(conv3_3)

    # Block 4
    conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(pool3)
    conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(conv4_1)
    conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(conv4_2)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool4')(conv4_3)

    # Block 5
    conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(pool4)
    conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(conv5_2)
    pool5 = MaxPooling2D((3, 3), strides=(1, 1), padding='same', name='pool5')(conv5_3)

    # FC6
    fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same', name='fc6')(pool5)

    # FC7
    fc7 = Conv2D(1024, (1, 1), activation='relu', padding='same', name='fc7')(fc6)

    # Block 6
    conv6_1 = Conv2D(256, (1, 1), activation='relu', padding='same', name='conv6_1')(fc7)
    conv6_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv6_padding')(conv6_1)
    conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', name='conv6_2')(conv6_1)

    # Block 7
    conv7_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv7_1')(conv6_2)
    conv7_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv7_padding')(conv7_1)
    conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid', name='conv7_2')(conv7_1)

    # Block 8
    conv8_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv8_1')(conv7_2)
    conv8_2 = Conv2D(256, (3, 3), activation='relu', padding='valid', name='conv8_2')(conv8_1)

    # Block 9
    conv9_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv9_1')(conv8_2)
    conv9_2 = Conv2D(256, (3, 3), activation='relu', padding='valid', name='conv9_2')(conv9_1)

    ############## SSD architecture ######################

    # Conv4_3
    
    conv4_3_norm = Normalize(gamma_init=20, name='conv4_3_norm')(conv4_3)
    num_prior = 4
    # Process Bounding box (x, y, h, w)
    # num_prior表示檢驗框, num_classes分類
    conv4_3_norm_mbox_loc = Conv2D(num_prior * 4, (3, 3), padding='same', name='conv4_3_norm_mbox_loc')(conv4_3_norm)
    conv4_3_norm_mbox_loc_flat = Flatten(name="conv4_3_norm_mbox_loc_flat")(conv4_3_norm_mbox_loc)

    conv4_3_norm_mbox_conf = Conv2D(num_prior * num_classes, (3, 3), padding='same', name='conv4_3_norm_mbox_conf')(conv4_3_norm)
    conv4_3_norm_mbox_conf_flat = Flatten(name='conv4_3_norm_mbox_conf_flat')(conv4_3_norm_mbox_conf)
    priorbox = PriorBox(img_size, 30.0, 60.0, aspect_ratios=[2],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv4_3_norm_mbox_priorbox')
    conv4_3_norm_mbox_priorbox = priorbox(conv4_3_norm)

    # FC7
    num_prior = 6
    fc7_mbox_loc = Conv2D(num_prior * 4, (3, 3), padding='same', name='fc7_mbox_loc')(fc7)
    fc7_mbox_loc_flat = Flatten(name="fc7_mbox_loc_flat")(fc7_mbox_loc)

    fc7_mbox_conf = Conv2D(num_prior * num_classes, (3, 3), padding='same', name='fc7_mbox_conf')(fc7)
    fc7_mbox_conf_flat = Flatten(name='fc7_mbox_conf_flat')(fc7_mbox_conf)
    priorbox = PriorBox(img_size, 60.0, 111.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='fc7_mbox_priorbox')
    fc7_mbox_priorbox = priorbox(fc7)

    # Conv6_2
    num_prior = 6
    conv6_2_mbox_loc = Conv2D(num_prior * 4, (3, 3), padding='same', name='conv6_2_mbox_loc')(conv6_2)
    conv6_2_mbox_loc_flat = Flatten(name='conv6_2_mbox_loc_flat')(conv6_2_mbox_loc)

    conv6_2_mbox_conf = Conv2D(num_prior * num_classes, (3, 3), padding='same', name='conv6_2_mbox_conf')(conv6_2)
    conv6_2_mbox_conf_flat = Flatten(name='conv6_2_mbox_conf_flat')(conv6_2_mbox_conf)

    priorbox = PriorBox(img_size, 111.0, 162.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv6_2_mbox_priorbox')
    conv6_2_mbox_priorbox = priorbox(conv6_2)

    # Conv7_2
    num_prior = 6
    conv7_2_mbox_loc = Conv2D(num_prior * 4, (3, 3), padding='same', name='conv7_2_mbox_loc')(conv7_2)
    conv7_2_mbox_loc_flat = Flatten(name='conv7_2_mbox_loc_flat')(conv7_2_mbox_loc)

    conv7_2_mbox_conf = Conv2D(num_prior * num_classes, (3, 3), padding='same', name='conv7_2_mbox_conf')(conv7_2)
    conv7_2_mbox_conf_flat = Flatten(name='conv7_2_mbox_conf_flat')(conv7_2_mbox_conf)

    priorbox = PriorBox(img_size, 162.0, 213.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv7_2_mbox_priorbox')
    conv7_2_mbox_priorbox = priorbox(conv7_2)

    # Conv8_2
    num_prior = 4
    conv8_2_mbox_loc = Conv2D(num_prior * 4, (3, 3), padding='same', name='conv8_2_mbox_loc')(conv8_2)
    conv8_2_mbox_loc_flat = Flatten(name='conv8_2_mbox_loc_flat')(conv8_2_mbox_loc)

    conv8_2_mbox_conf = Conv2D(num_prior * num_classes, (3, 3), padding='same', name='conv8_2_mbox_conf')(conv8_2)
    conv8_2_mbox_conf_flat = Flatten(name='conv8_2_mbox_conf_flat')(conv8_2_mbox_conf)

    priorbox = PriorBox(img_size, 213.0, 264.0, aspect_ratios=[2],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv8_2_mbox_priorbox')
    conv8_2_mbox_priorbox = priorbox(conv8_2)

    # Conv9_2
    num_prior = 4
    conv9_2_mbox_loc = Conv2D(num_prior * 4, (3, 3), padding='same', name='conv9_2_mbox_loc')(conv9_2)
    conv9_2_mbox_loc_flat = Flatten(name='conv9_2_mbox_loc_flat')(conv9_2_mbox_loc)

    conv9_2_mbox_conf = Conv2D(num_prior * num_classes, (3, 3), padding='same', name='conv9_2_mbox_conf')(conv9_2)
    conv9_2_mbox_conf_flat = Flatten(name='conv9_2_mbox_conf_flat')(conv9_2_mbox_conf)

    priorbox = PriorBox(img_size, 264.0, 315.0, aspect_ratios=[2],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv9_2_mbox_priorbox')
    conv9_2_mbox_priorbox = priorbox(conv9_2)


    mbox_loc = concatenate([conv4_3_norm_mbox_loc_flat,
                            fc7_mbox_loc_flat,
                            conv6_2_mbox_loc_flat,
                            conv7_2_mbox_loc_flat,
                            conv8_2_mbox_loc_flat,
                            conv9_2_mbox_loc_flat], axis=1, name="mbox_loc")

    mbox_conf = concatenate([conv4_3_norm_mbox_conf_flat,
                            fc7_mbox_conf_flat,
                            conv6_2_mbox_conf_flat,
                            conv7_2_mbox_conf_flat,
                            conv8_2_mbox_conf_flat,
                            conv9_2_mbox_conf_flat], axis=1, name="mbox_conf")

    mbox_priorbox = concatenate([conv4_3_norm_mbox_priorbox,
                                fc7_mbox_priorbox,
                                conv6_2_mbox_priorbox,
                                conv7_2_mbox_priorbox,
                                conv8_2_mbox_priorbox,
                                conv9_2_mbox_priorbox], axis=1, name="mbox_priorbox")
    

    num_boxes = mbox_loc.get_shape().as_list()[-1] // 4
    
    '''
    print(mbox_loc.get_shape().as_list()[-1])
    if hasattr(mbox_loc, "_keras_shape"):
        num_boxes = mbox_loc._keras_shape[-1] // 4
    elif hasattr(mbox_loc, 'int_shape'):
        num_boxes = K.int_shape(mbox_loc)[-1] // 4
    '''
    mbox_loc = Reshape((num_boxes, 4), name='mbox_loc_final')(mbox_loc)
    mbox_conf = Reshape((num_boxes, num_classes), name='mbox_conf_logits')(mbox_conf)
    mbox_conf = Activation('softmax', name='mbox_conf_final')(mbox_conf)

    predictions = concatenate([mbox_loc,
                            mbox_conf,
                            mbox_priorbox], axis=2, name='predictions')
    
    print(predictions)
    model = Model(input_tensor, predictions)
    return model


