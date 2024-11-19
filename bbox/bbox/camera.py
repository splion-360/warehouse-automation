

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class MinimalVideoPublisher(Node):
    def __init__(self, name = "video"):
        super().__init__(name)
        rclpy.logging.set_logger_level('semantic_segmentation', rclpy.logging.LoggingSeverity.INFO)


        ## Build a subscriber node to Image stream
        self.cv_bridge = CvBridge()
        self.image_publisher = self.create_publisher(Image, '/camera/image_raw', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.cap = cv2.VideoCapture(0)  ## Video stream from the desktop camera
        self.get_user_input = 0
    
    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret: 
            img_msg = self.cv_bridge.cv2_to_imgmsg(frame, 'bgr8')
            self.image_publisher.publish(img_msg)
            self.get_logger().info("published video frame")
            self.get_user_input = cv2.waitKey(10)
    


def main(args = None):
    rclpy.init(args = args)
    video = MinimalVideoPublisher()
    
    while rclpy.ok():
        try:
            rclpy.spin_once(video) # Trigger callback processing.		
        except KeyboardInterrupt as e: 
            break

    video.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()