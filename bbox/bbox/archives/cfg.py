import torch 

WEIGHT_PATH = "src/bbox/bbox/weights"
DEVICE = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
IMG_SIZE = (448, 448)
CONF_THRESH, IOU_THRESH = 0.25, 0.45
CLASSES = [0]
NMS = False
MAX_DET = 1000 
LINE_WIDTH = 3
