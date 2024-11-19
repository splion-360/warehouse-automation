import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
import numpy as np
import torch
import time
from seg.model import SegNet
from seg.utils import get_transforms
from seg.cfg import *

class ObjectSegmentation(Node):
    def __init__(self, name = "semantic_segmentation"):
        super().__init__(name)
        rclpy.logging.set_logger_level('semantic_segmentation', rclpy.logging.LoggingSeverity.INFO)

        self.weights = WEIGHT_PATH
        self.device = DEVICE
    
        ## Step 1: Model initialisation and model loading 
        self.model = SegNet(IMG['IN_CHANNELS'], IMG['OUT_CHANNELS'])
        self.load_model(self.weights)
        self.model.to(self.device)

        ## Build a subscriber node to Image stream
        self.cv_bridge = CvBridge()
        self.image_subscriber = self.create_subscription(Image, 
                                                    '/camera/image_raw',
                                                    self.image_callback, 
                                                    10)

        self.image_publisher_viz =  self.create_publisher(Image, 
                                                        '/segment/viz',
                                                        10)
        self.image_subscriber
    
    
    def image_callback(self, img : Image):
        img_BGR = self.cv_bridge.imgmsg_to_cv2(img, "bgr8")
        
        if img_BGR is not None: 
            self.get_logger().info("image received and passed to the model for prediction !!!")
            
            start = time.time()
            mask = self.infer(img_BGR)
            end = time.time() - start
            
            self.get_logger().info(("mask obtained and published | inference time : %.3f ms") % (end * 1000))
            detection_img = self.cv_bridge.cv2_to_imgmsg(mask, 'mono8')
            self.image_publisher_viz.publish(detection_img)

    def load_model(self, weight_path):
        self.get_logger().info('Loading model weights from {}'.format(weight_path))
        last_weight = os.path.join(weight_path, "best.pt")
        chkpt = torch.load(last_weight, map_location = self.device)
        self.model.load_state_dict(chkpt['model'])
        del chkpt
    
    def infer(self, img : np.ndarray) -> np.ndarray:
        self.model.eval()
        _, test_transform = get_transforms(IMG['HEIGHT'], IMG['WIDTH'], **{'flip' : 0.5})
        transformed_img = test_transform(image = img)["image"][None,...]
        transformed_img = transformed_img.to(self.device)

        with torch.no_grad():        
            pred = torch.sigmoid(self.model(transformed_img))
            pred_mask = (pred > 0.5) * 255
            pred_mask = np.squeeze(pred_mask.cpu().numpy().astype(np.uint8))

        return pred_mask


def main(args = None):
    rclpy.init(args = args)
    segmenter = ObjectSegmentation()
    
    while rclpy.ok():
        try:
            rclpy.spin_once(segmenter) # Trigger callback processing.		
        except KeyboardInterrupt as e: 
            break

    segmenter.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()