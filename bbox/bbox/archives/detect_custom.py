import torch
import bbox.gpu as gpu
from bbox.model.yolov3 import Yolov3
from bbox.tools import *
import os
import bbox.yolov3_config_voc as cfg
from bbox.visualize import *
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time
import cv2

class ObjectDetection(Node):
    def __init__(self, name = "bounding_box"):
        super().__init__(name)
        self.img_size = cfg.DATA['IMG_SIZE']
        self.num_class = cfg.DATA["NUM"]
        self.conf_thresh = cfg.TEST["CONF_THRESH"]
        self.nms_thresh = cfg.TEST["NMS_THRESH"]
        self.device = gpu.select_device(0)
        self.classes = cfg.DATA["CLASSES"]
        self.weight_path = cfg.DATA["WEIGHT_PATH"]

        self.model = Yolov3(cfg).to(self.device)
        self.load_model_weights(self.weight_path)

        ## Build a subscriber node to Image stream
        self.cv_bridge = CvBridge()
        self.image_subscriber = self.create_subscription(Image, 
                                                        '/camera/image_raw',
                                                        self.image_callback, 
                                                        10)

        self.image_publisher_viz =  self.create_publisher(Image, 
                                                        '/bbox/viz',
                                                        10)
        




    def load_model_weights(self, weight_path):
        print("Loading weight file from : {}".format(weight_path))

        weight = os.path.join(weight_path)
        chkpt = torch.load(weight, map_location=self.device)
        self.model.load_state_dict(chkpt)
        del chkpt
    

    def get_bbox(self, img):
        bboxes = self.predict(img, self.img_size, (0, np.inf)) 
        bboxes = nms(bboxes, self.conf_thresh, self.nms_thresh)
        return bboxes

    def predict(self, img, test_shape, valid_scale):
        org_img = np.copy(img)
        org_h, org_w, _ = org_img.shape

        img = self.get_img_tensor(img, test_shape).to(self.device)
        self.model.eval()
        with torch.no_grad():
            _, p_d = self.model(img)
        pred_bbox = p_d.squeeze().cpu().numpy()
        bboxes = self.convert_pred(pred_bbox, test_shape, (org_h, org_w), valid_scale)
        return bboxes
    
    def get_img_tensor(self, img, test_shape):
        img = self.resize(img, (test_shape, test_shape)).transpose(2, 0, 1)
        return torch.from_numpy(img[np.newaxis, ...]).float()
    
    def resize(self, img, target_shape): 
        h_org , w_org , _= img.shape
        h_target, w_target = target_shape

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        resize_ratio = min(1.0 * w_target / w_org, 1.0 * h_target / h_org)
        resize_w = int(resize_ratio * w_org)
        resize_h = int(resize_ratio * h_org)
        image_resized = cv2.resize(img, (resize_w, resize_h))

        image_paded = np.full((h_target, w_target, 3), 128.0)
        dw = int((w_target - resize_w) / 2)
        dh = int((h_target - resize_h) / 2)
        image_paded[dh:resize_h + dh, dw:resize_w + dw, :] = image_resized
        image = image_paded / 255.0  
        return image


    def convert_pred(self, pred_bbox, test_input_size, org_img_shape, valid_scale):
     
        pred_coor = xywh2xyxy(pred_bbox[:, :4])
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]

        org_h, org_w = org_img_shape
        resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
        dw = (test_input_size - resize_ratio * org_w) / 2
        dh = (test_input_size - resize_ratio * org_h) / 2
        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio


        pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                    np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)

        invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
        pred_coor[invalid_mask] = 0


        bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
        scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))


        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        score_mask = scores > self.conf_thresh

        mask = np.logical_and(scale_mask, score_mask)

        coors = pred_coor[mask]
        scores = scores[mask]
        classes = classes[mask]

        bboxes = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

        return bboxes

    
    def infer(self, img):
        bboxes = self.get_bbox(img)
        if bboxes.shape[0] != 0:
            bbox = bboxes[...,:4]
            class_inds = bboxes[...,5].astype(np.int32)
            class_probs = bboxes[..., 4]
            visualize_boxes(img, bbox, class_inds, class_probs, self.classes)
            return img 
        return None

    def image_callback(self, img):
        img_BGR = self.cv_bridge.imgmsg_to_cv2(img, "bgr8")
        
        if img_BGR is not None: 
            self.get_logger().info("image received and passed to the model for prediction !!!")
            
            start = time.time()
            pred = self.infer(img_BGR)
            end = time.time() - start
            
            self.get_logger().info(("bounding box annotated obtained and published | inference time : %.3f ms") % (end * 1000))
            detection_img = self.cv_bridge.cv2_to_imgmsg(pred, 'bgr8')
            self.image_publisher_viz.publish(detection_img)        
        
   
def main(args = None):
    rclpy.init(args = args)
    node = ObjectDetection()
    
    while rclpy.ok():
        try:
            rclpy.spin_once(node) # Trigger callback processing.		
        except KeyboardInterrupt as e: 
            break

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
