import numpy as np
import tensorflow as tf
from PIL import Image

def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new("RGB", size, (128, 128, 128))
    new_image.paste(image, ((w-nw)//2, (h-nw)//2))
    x_offset = ((w-nw) // 2) / 300
    y_offset = ((h-nh) // 2) / 300
    return new_image, x_offset, y_offset

def ssd_correct_boxes(top, left, bottom, right, input_shape, image_shape):
    
    new_shape = image_shape*np.min(input_shape/image_shape)

    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape

    box_yx = np.concatenate(((top+bottom)/2,(left+right)/2),axis=-1)
    box_hw = np.concatenate((bottom-top,right-left),axis=-1)

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  np.concatenate([
        box_mins[:, 0:1],
        box_mins[:, 1:2],
        box_maxes[:, 0:1],
        box_maxes[:, 1:2]
    ],axis=-1)
    boxes *= np.concatenate([image_shape, image_shape],axis=-1)
    return boxes
class BBoxUtility(object):

    def __init__(self, num_classes, priors=None, overlap_threshold=0.5, nms_thresh=0.45, top_k=400):

        self.num_classes = num_classes
        self.priors = priors
        self.num_priors = 0 if priors is None else len(priors)
        self.overlap_threshold = overlap_threshold
        self._nms_thresh = nms_thresh
        self._top_k = top_k
        self.boxes = tf.placeholder(dtype='float32', shape=(None, 4))
        self.scores = tf.placeholder(dtype="float32", shape=(None,))
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores, self._top_k, iou_threshold=self._nms_thresh)

        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU':0}))
    
    @property
    def nms_thresh(self):
        return self._nms_thresh
    
    @nms_thresh.setter
    def nms_thresh(self, value):
        self._nms_thresh = value
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._top_k, iou_threshold=self._nms_thresh)
    
    @property
    def top_k(self):
        return self._top_k
    
    @top_k.setter
    def top_k(self, value):
        self._top_k = value
        self.nms = tf.image.non_max_suppression(self.boxes, self.scores,
                                                self._top_k, iou_threshold=self._nms_thresh)

    def iou(self, box):
        # 計算每個框的iou

        inter_min = np.maximum(self.priors[:, :2], box[:2])
        inter_max = np.minimum(self.priors[:, 2:4], box[2:])

        inter_wh = inter_max - inter_min
        inter_wh = np.maximum(inter_wh, 0)
        
        # inter area
        inter_area = inter_wh[:, 0] * inter_wh[:, 1]
        # true bounding box area
        true_area = (box[2] - box[0]) * (box[3] - box[1])
        # predict bounding box area
        predict_area = (self.priors[:, 2] - self.priors[:, 1]) * (self.priors[:, 3] - self.priors[:, 1])

        # Compute iou
        union = true_area + predict_area - inter_area

        iou = inter_area / union
        return iou

    def encode_box(self, box, return_iou=True):

        iou = self.iou(box)
        encode_box = np.zeros((self.num_priors, 4 + return_iou))

        # 重和度較高的預測框
        assign_mask = iou > self.overlap_threshold
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        if return_iou:
            encode_box[:, -1][assign_mask] = iou[assign_mask]

        assign_priors = self.priors[assign_mask]

        #計算真實框的中心與長寬
        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]
        #計算重合度高的預測框中心與長寬
        
        assign_priors_center = 0.5 * (assign_priors[:, :2] + assign_priors[:, 2:4])
        assign_priors_wh = (assign_priors[:, 2:4] - assign_priors[:, :2])

        # 求取SSD預測結果
        encode_box[:, :2][assign_mask] = box_center - assign_priors_center
        encode_box[:, :2][assign_mask] /= assign_priors_wh
        encode_box[:, :2][assign_mask] /= assign_priors[:, -4:-2]

        encode_box[:, 2:4][assign_mask] = np.log(box_wh / assign_priors_wh)
        encode_box[:, 2:4][assign_mask] /= assign_priors[:, -2:]
        return encode_box.ravel()
    
    def assign_boxes(self, boxes):

        assignment = np.zeros((self.num_priors, 4 + self.num_classes + 8))
        assignment[:, 4] = 1.0
        if len(boxes) == 0:
            return assignment
        
        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])
        encoded_boxes = encoded_boxes.reshape(-1, self.num_priors, 5)

        # 取重合度最高的預測框
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]

        assign_num = len(best_iou_idx)
        # 保留重和度最大的預測框
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), :4]
        # 4 代表為背景概率
        assignment[:, 4][best_iou_mask] = 0
        assignment[:, 5:-8][best_iou_mask] = boxes[best_iou_idx, 4:]
        assignment[:, -8][best_iou_mask] = 1
        return assignment
    
    def decode_boxes(self, mbox_loc, mbox_priorbox, variances):

        # 獲取預測框的center_x, center_y, w, h
        prior_w = mbox_priorbox[:, 2] - mbox_priorbox[:, 0]
        prior_h = mbox_priorbox[:, 3] - mbox_priorbox[:, 1]
        prior_x = 0.5 * (mbox_priorbox[:, 0] + mbox_priorbox[:, 2])
        prior_y = 0.5 * (mbox_priorbox[:, 1] + mbox_priorbox[:, 3])

        # 預測框與實際框中心的偏移差距
        decoded_bbox_x = mbox_loc[:, 0] * prior_w * variances[:, 0]
        decoded_bbox_x += prior_x
        decoded_bbox_y = mbox_loc[:, 1] * prior_h * variances[:, 1]
        decoded_bbox_y += prior_y

        # 真實框的寬與高
        decoded_bbox_w = np.exp(mbox_loc[:, 2] * variances[:, 2])
        decoded_bbox_w *= prior_w
        decoded_bbox_h = np.exp(mbox_loc[:, 3] * variances[:, 3])
        decoded_bbox_h *= prior_h

        # 計算 xmin, ymin, xmax, ymax
        xmin = decoded_bbox_x - (0.5 * decoded_bbox_w)
        ymin = decoded_bbox_y - (0.5 * decoded_bbox_h)
        xmax = decoded_bbox_x + (0.5 * decoded_bbox_w)
        ymax = decoded_bbox_y + (0.5 * decoded_bbox_h)

        bbox = np.concatenate((xmin[:, None],
                            ymin[:, None],
                            xmax[:, None],
                            ymax[:, None]), axis=-1)

        bbox = np.minimum(np.maximum(bbox, 0.0), 1.0)
        return bbox

    def detection_out(self, predictions, background_label=0, keep_top_k=200, conf_threshold=0.5):

        mbox_loc = predictions[:, :, :4]
        variances = predictions[:, :, -4:]
        mbox_priorbox = predictions[:, :, -8:-4]
        mbox_conf = predictions[:, :, 4:-8]
        results = []

        for i in range(len(mbox_loc)):
            results.append([])
            decode_bbox = self.decode_boxes(mbox_loc[i], mbox_priorbox[i], variances[i])

            for c in range(self.num_classes):
                if c == background_label:
                    continue
                c_confs = mbox_conf[i, :, c]
                c_confs_m = c_confs > conf_threshold
                if len(c_confs[c_confs_m]) > 0:
                    boxes_to_process = decode_bbox[c_confs_m]
                    confs_to_process = c_confs[c_confs_m]
                    
                    # NMS
                    feed_dict = {self.boxes: boxes_to_process,
                                self.scores: confs_to_process}
                    idx = self.sess.run(self.nms, feed_dict=feed_dict)

                    good_boxes = boxes_to_process[idx]
                    confs = confs_to_process[idx][:, None]
                    labels = c * np.ones((len(idx), 1))
                    c_pred = np.concatenate((labels, confs, good_boxes), axis=1)
                    results[-1].extend(c_pred)
            
            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])
                argsort = np.argsort(results[-1][:, 1])[::-1]
                results[-1] = results[-1][argsort]
                results[-1] = results[-1][:keep_top_k]
        
        return results
