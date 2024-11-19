import sys
sys.path.append("..")

import torch.nn as nn
import torch
from model.backbones.darknet53 import Darknet53
from model.necks.yolo_fpn import FPN_YOLOV3
from model.head.yolo_head import Yolo_head
# from student_submission.yolo_head import Yolo_head
from model.layers.conv_module import Convolutional
import config.yolov3_config_voc as cfg
import numpy as np
from utils.tools import *


class Yolov3(nn.Module):
    """
    Note ： int the __init__(), to define the modules should be in order, because of the weight file is order
    Comprises the entire YoloV3 architecture - darknet53 backbone, Feature Pyramid Network (FPN), and Yolo heads

    1.) Darknet53 backbone: Computes multi-scale feature outputs having different channel dimensions [1024, 512, 256]
    Output = [N, 1024, x_s, x_s], [N, 512, x_m, x_m], [N, 256, x_l, x_l]
    Here, N -> batch_size and x_s,x_m,x_l are small, medium and large spatial dimensions.
    x_l = 2*x_m, x_m=2*x_s
    NOTE: Darknet53() already provided

    2.) FPN: Takes backbone's output and computes features at 3 spatial scales using bottom-up pathway in pyramid network. All features have same channel dimensions but different spatial scales.
    Output = [N, C', x_s, x_s], [N, C', x_m, x_m], [N, C', x_l, x_l]
    Here, N -> batch_size, 
    C' = Number of anchors*C  , where C= no. of classes + 4 channels (x,y,w,h) + 1 channel (box confidence)
    NOTE: FPN_YOLOV3() already provided

    3.) Yolo Heads: Computes final Image-scale predictions (box coordinates, box confidence and class probabilities) from the FPN output using anchor boxes and strides.
    NOTE: Refer Yolo_head() class for more details
    Output = (p, p_d)
    Here, p -> Reshaped FPN predictions (from Yolo_head forward()) on a spatial scale
    p_d -> Final image-scale predictions (from Yolo_head forward()) on the same spatial scale
    
    FINAL OUTPUT = (p, p_d)
    Here, p is tuple of reshaped FPN predictions (from Yolo_head forward()) on all 3 spatial scales (small, medium and large). 
    p_d is tuple of Final image-scale predictions (from Yolo_head forward()) on same 3 spatial scales as p's.
    
    Refer original YoloV3 paper: "YoloV3 - an incremental improvement": https://arxiv.org/pdf/1804.02767.pdf for more details on converting offset predictions into original image-scale predictions
    Refer vanilla Yolo paper: "You Only Look Once: Unified, Real-Time Object Detection": https://arxiv.org/pdf/1506.02640.pdf

    TODO:  Initialize FPN_YOLOV3 and Yolo_heads. And Complete the Forward Pass
    """
    def __init__(self, cfg, init_weights=True):
        super().__init__()

        self.__anchors = torch.FloatTensor(cfg.MODEL["ANCHORS"])                # anchor boxes for all 3 spatial scales
        self.__strides = torch.FloatTensor(cfg.MODEL["STRIDES"])                # strides for all 3 spatial scales
        self.__nC = cfg.DATA["NUM"]                                             # number of classes
        self.__out_channel = cfg.MODEL["ANCHORS_PER_SCLAE"] * (self.__nC + 5) 

        self.__backnone = Darknet53() # Darknet53 backbone

        self.__fpn = FPN_YOLOV3((1024,512,256),(self.__out_channel,self.__out_channel,self.__out_channel))     # Feature Pyramid Network (FPN)
        self.__head_s = Yolo_head(self.__nC,self.__anchors[0],self.__strides[0])  # small yolo head
        self.__head_m = Yolo_head(self.__nC,self.__anchors[1],self.__strides[1])  # medium yolo head
        self.__head_l = Yolo_head(self.__nC,self.__anchors[2],self.__strides[2])  # large yolo head

        # TODO: Intialize the FPN and Yolo Heads here #
        # Go to their respective classes FPN_YOLOV3() and Yolo_head() for more details about initializing them.
        ### START YOUR CODE HERE ###

        ### END YOUR CODE ###

        if init_weights:
            self.__init_weights()


    def forward(self, x):
        """
        TODO: Complete the forward pass
        """
        out = []

        x_s, x_m, x_l = self.__backnone(x)
        x_s,x_m,x_l = self.__fpn(x_l,x_m,x_s)
        p_s,p_d_s = self.__head_s(x_s)
        p_m,p_d_m = self.__head_m(x_m)
        p_l,p_d_l = self.__head_l(x_l)
        p,p_d = (p_s,p_m,p_l),(p_d_s,p_d_m,p_d_l)
        
        if self.training:
            return p, p_d
        else:
            p_d_s = p_d[0].reshape(-1, 5 + self.__nC)
            p_d_m = p_d[1].reshape(-1, 5 + self.__nC)
            p_d_l = p_d[2].reshape(-1, 5 + self.__nC)
            p_d_final = (p_d_s, p_d_m, p_d_l)
            return p, torch.cat(p_d_final, 0)

        ### START YOUR CODE HERE ###

        ### END YOUR CODE HERE ###

    def __init_weights(self):

        " Note ：nn.Conv2d nn.BatchNorm2d'initing modes are uniform "
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                print("initing {}".format(m))

            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight.data, 1.0)
                torch.nn.init.constant_(m.bias.data, 0.0)

                print("initing {}".format(m))


    def load_darknet_weights(self, weight_file, cutoff=52):
        "https://github.com/ultralytics/yolov3/blob/master/models.py"

        print("load darknet weights : ", weight_file)

        with open(weight_file, 'rb') as f:
            _ = np.fromfile(f, dtype=np.int32, count=5)
            weights = np.fromfile(f, dtype=np.float32)
        count = 0
        ptr = 0
        for m in self.modules():
            if isinstance(m, Convolutional):
                # only initing backbone conv's weights
                if count == cutoff:
                    break
                count += 1

                conv_layer = m._Convolutional__conv
                if m.norm == "bn":
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = m._Convolutional__norm
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias.data)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight.data)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b

                    print("loading weight {}".format(bn_layer))
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias.data)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight.data)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

                print("loading weight {}".format(conv_layer))
