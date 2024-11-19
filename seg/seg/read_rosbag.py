import rclpy
from rclpy.node import Node
import rosbag2_py
import yaml
import os
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from seg.model import SegNet
from seg.utils import get_transforms
import seg.cfg as cfg
import torch 
import time


## Get the path info from yaml 
def load_config(cfg_file : str):
    with open(cfg_file, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


class Segmentation(Node):

    def __init__(self, name = 'seg_bag_reader'):
        super().__init__(name)

        rclpy.logging.set_logger_level('seg_bag_reader', rclpy.logging.LoggingSeverity.INFO)
        self.config = load_config(os.path.join(cfg.ROOT, 'bag/metadata.yaml'))['rosbag2_bagfile_information']
        self.reader = rosbag2_py.SequentialReader()
        file_path = os.path.join(cfg.ROOT, self.config['files']['path'])

        storage_options = rosbag2_py.StorageOptions(uri = file_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(input_serialization_format = 'cdr', output_serialization_format = 'cdr')
        self.reader.open(storage_options, converter_options)
        self.topic_of_interest = self.config['rosbag2_bagfile_information']['topics_with_message_count'][1]['topic_metadata']['name']
        self.type = self.config['rosbag2_bagfile_information']['topics_with_message_count'][1]['topic_metadata']['type']
        self.timer = self.create_timer(1, self.timer_callback)

        self.weights = os.path.join(cfg.ROOT, 'weights')
        self.device = cfg.DEVICE
    
        ## Step 1: Model initialisation and model loading 
        self.model = SegNet(cfg.IMG['IN_CHANNELS'], cfg.IMG['OUT_CHANNELS'])
        self.load_model(self.weights)
        self.model.to(self.device)

        ## Build a publisher node to display the mask
        self.cv_bridge = CvBridge()
        self.image_publisher_viz =  self.create_publisher(Image, 
                                                        '/segment/viz',
                                                        10)
        
    
    def load_model(self, weight_path):
        self.get_logger().info('Loading model weights from {}'.format(weight_path))
        last_weight = os.path.join(weight_path, "best.pt")
        chkpt = torch.load(last_weight, map_location = self.device)
        self.model.load_state_dict(chkpt['model'])
        del chkpt
    
    def infer(self, img : np.ndarray) -> np.ndarray:
        self.model.eval()
        _, test_transform = get_transforms(cfg.IMG['HEIGHT'], cfg.IMG['WIDTH'], **{'flip' : 0.5})
        transformed_img = test_transform(image = img)["image"][None,...]
        transformed_img = transformed_img.to(self.device)

        with torch.no_grad():        
            pred = torch.sigmoid(self.model(transformed_img))
            pred_mask = (pred > 0.5) * 255
            pred_mask = np.squeeze(pred_mask.cpu().numpy().astype(np.uint8))

        return pred_mask



    def timer_callback(self):
        while self.reader.has_next():
            topic, data, t = self.reader.read_next()
            self.get_logger().info('msg received from the ros bag file')
            
            if topic == self.topic_of_interest:
                msg = self.cv_bridge.deserialize(data, self.type)
                img_BGR = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
                self.get_logger().info("image received and passed to the model for prediction !!!")
            
                start = time.time()
                mask = self.infer(img_BGR)
                end = time.time() - start
            
                self.get_logger().info(("mask obtained and published | inference time : %.3f ms") % (end * 1000))
                detection_img = self.cv_bridge.cv2_to_imgmsg(mask, 'mono8')
                self.image_publisher_viz.publish(detection_img)

            


def main(args = None):
    rclpy.init(args = args)
    segmenter = Segmentation()
    
    while rclpy.ok():
        try:
            rclpy.spin_once(segmenter) # Trigger callback processing.		
        except KeyboardInterrupt as e: 
            break

    segmenter.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()