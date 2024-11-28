import rclpy
from rclpy.node import Node
import sensor_msgs.msg as msg
import cv2
from cv_bridge import CvBridge
import numpy as np
import torch
import time
import yaml
import os
from bbox.models.common import DetectMultiBackend
from bbox.utils.general import non_max_suppression, scale_boxes
from bbox.utils.augmentations import letterbox
import bbox.yolov3_config_voc as cfg
from bbox.visualize import visualize_boxes


## Get the path info from yaml 
def load_config(cfg_file : str):
    with open(cfg_file, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


class ObjectDetection(Node):
    def __init__(self, name = 'bounding_box'):
        super().__init__(name)
        rclpy.logging.set_logger_level(name, rclpy.logging.LoggingSeverity.INFO)
        self.weights = os.path.join(cfg.ROOT, 'weights', 'best.pt')
        self.device  = cfg.DATA["DEVICE"]
        self.imgsz   = cfg.DATA["IMG_SIZE"]
        self.conf    = cfg.TEST["CONF_THRESH"]
        self.iou_thresh = cfg.TEST["IOU_THRESH"]
        self.classes = cfg.DATA["CLASSES"]
        self.agnostic_nms = False
        self.max_det = cfg.TEST["MAX_DET"]

        topic_of_interest = "/robot1/zed2i/left/image_rect_color"
        self.get_logger().info(topic_of_interest)
        self.image_subscriber = self.create_subscription(msg.Image, 
                                                    topic_of_interest,
                                                    self.image_callback, 
                                                    10)
        
    
        ## Step 1: Model initialisation and model loading 
        self.model = DetectMultiBackend(self.weights, self.device)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        ## Build a subscriber node to Image stream
        self.cv_bridge = CvBridge()
     

        self.image_publisher_viz =  self.create_publisher(msg.Image, 
                                                        '/bbox/viz',
                                                        10)
        

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
        

    def infer(self, img):
        im = letterbox(img, self.imgsz, stride=self.stride, auto=True)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im).to(self.model.device)
        im = im.float()  
        im /= 255  
        if len(im.shape) == 3:
            im = im[None] 

        pred = self.model(im)
        pred = non_max_suppression(pred, self.conf, self.iou_thresh, [i for i in range(len(self.classes))], self.agnostic_nms, self.max_det)
        if pred is not None: 
            pred = pred[0].cpu().numpy()
            bbox = pred[...,:4]
            bbox = scale_boxes(im.shape[2:], bbox, img.shape).round()
            class_inds = pred[...,5].astype(np.int32)
            class_probs = pred[..., 4]
            visualize_boxes(img, bbox, class_inds, class_probs, self.classes)
   
        return img
        

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