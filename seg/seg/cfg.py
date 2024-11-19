import torch


ROOT = "src/seg/seg/"
IMG = {"IN_CHANNELS" : 3, 
       "OUT_CHANNELS": 1,
       "HEIGHT" : 416,
       "WIDTH": 416}
DEVICE = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 1

