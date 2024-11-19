# coding=utf-8
# project
import os
import torch
DATA_PATH = os.getcwd()
PROJECT_PATH = os.getcwd()

DATA = {"CLASSES":['pallet'],
        "NUM":1,
        "IMG_SIZE": 416,
        "DEVICE" : torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu'),
        "WEIGHT_PATH":"src/bbox/bbox/weights/best.pt"}
# model
MODEL = {"ANCHORS":[[(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)],  # Anchors for small obj
            [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],  # Anchors for medium obj
            [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]] ,# Anchors for big obj
         "STRIDES":[8, 16, 32],
         "ANCHORS_PER_SCLAE":3
         }


# test
TEST = {
        "TEST_IMG_SIZE":416,
        "BATCH_SIZE":1,
        "NUMBER_WORKERS":0,
        "CONF_THRESH":0.1,
        "IOU_THRESH":0.5,
        "MAX_DET":1000
        }
