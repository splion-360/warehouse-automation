import cv2
import json
import os
import numpy as np
import typing
from albumentations.pytorch import ToTensorV2
import albumentations as A
import torch
import warnings
warnings.filterwarnings('ignore')

class Metrics: 
    def __init__(self, model, iou_thresh : typing.List, device : str = 'cuda'): 
        self.threshold = torch.arange(iou_thresh[0], iou_thresh[1], iou_thresh[2])
        self.EPS = 1e-07
        self.device = device
        self.model = model


    def __call__(self, loader, criterion) -> typing.Tuple[float, float, float, float, float]:
        """
        Calculate the accuracy, dice, IOU and mAP of a binary segmentation model on a given dataloader
    
        """ 
        
        N = len(loader)
        self.model.eval()
        accuracy, dice, mAP, iou = 0, 0, 0, 0
        loss = 0
        for i ,data in enumerate(loader):
            img, gt_mask = data
            img, gt_mask = img.to(self.device), gt_mask.to(self.device)
            pred = self.model(img)
            loss += criterion(pred, gt_mask).item()
            pred_mask = torch.sigmoid(pred) > 0.5

            accuracy += self.get_accuracy(gt_mask, pred_mask)
            dice     += self.get_dice(gt_mask, pred_mask)
            iou      += self.get_iou(gt_mask, pred_mask)
            mAP      += self.get_mAP(gt_mask, pred_mask)

        self.model.train()
        return loss / N, accuracy / N, dice / N, iou / N, mAP / N


    def get_iou(self, gt_mask, pred_mask):
        """
        Computes area of intersection / area of union using the provided binary masks
        """
        intersection = torch.logical_and(pred_mask, gt_mask).sum()
        union = torch.logical_or(pred_mask, gt_mask).sum()
        return intersection / (union + self.EPS)
    
    def compute_ap(self, gt_mask, pred_mask):
        """
        Computes the precision and recall at different IOU thresholds and returns an array 
        """

        
        fp, tp, fn = torch.zeros(len(self.threshold)), torch.zeros(len(self.threshold)), torch.zeros(len(self.threshold))

        for i, iou_threshold in enumerate(self.threshold):  
            iou = self.get_iou(gt_mask, pred_mask)

            if iou >= iou_threshold: 
                tp[i] = 1
            else:
                if torch.sum(pred_mask) > 0:
                    fp[i] = 1
                if torch.sum(gt_mask) > 0: 
                    fn[i] = 1

        precision = tp / (tp + fp + self.EPS)
        recall = tp / (tp + fn + self.EPS)

        return precision, recall 

    def average_precision(self, precision : torch.tensor , recall : torch.tensor):
        """
        Calculates the area under the ROC curve (AUC) for a given precision and recall array
        
        """
        i = torch.argsort(recall)
        sorted_precision, sorted_recall = torch.zeros(precision.shape[0] + 2), torch.zeros(recall.shape[0] + 2)
        sorted_precision[1:-1] = precision[i] 
        sorted_recall[1:-1]    = recall[i]
        sorted_precision[0], sorted_recall[0] = 1, 0
        sorted_precision[-1], sorted_recall[-1] = 0, 1
        
        for i in range(len(sorted_precision) - 2, -1, -1):
            sorted_precision[i] = max(sorted_precision[i], sorted_precision[i + 1])
              
        ap = torch.trapz(sorted_precision, sorted_recall)

        return ap
    
    def get_mAP(self, gt_mask, pred_mask):

        """
        Calculates the mean average precision of a binary segmentation mask
        """
        
        precision, recall = self.compute_ap(gt_mask, pred_mask)
        ap = self.average_precision(precision, recall)
        return ap
    
    def get_accuracy(self, gt_mask, pred_mask):
        num = torch.sum(pred_mask == gt_mask)
        den = gt_mask.shape.numel()
        return num / (den + self.EPS)
    
    def get_dice(self, gt_mask, pred_mask):
        intersection = torch.sum(pred_mask[gt_mask == 1])
        numels = torch.sum(pred_mask + gt_mask)
        return 2 * (intersection / (numels + self.EPS))
    


class CosineDecayLR(object):

    def __init__(self, optimizer, T_max, lr_init, lr_min = 0., warmup = 0):
        super().__init__()
        self.__optimizer = optimizer
        self.__T_max = T_max
        self.__lr_min = lr_min
        self.__lr_max = lr_init
        self.__warmup = warmup


    def step(self, t):
        if self.__warmup and t < self.__warmup:
            lr = self.__lr_max / self.__warmup * t
        else:
            T_max = self.__T_max - self.__warmup
            t = t - self.__warmup
            lr = self.__lr_min + 0.5 * (self.__lr_max - self.__lr_min) * (1 + np.cos(t/T_max * np.pi))
        for param_group in self.__optimizer.param_groups:
            param_group["lr"] = lr



def get_transforms(img_height : int = 320, img_width : int = 480, **kwargs) -> typing.Tuple:

    train_transform = A.Compose([   A.Resize(img_height, img_width),
                                    A.Normalize(mean=(0,0,0),std=(1,1,1)), 
                                    A.HorizontalFlip(kwargs['flip']),
                                    A.RandomBrightnessContrast(p = 0.7),
                                    ToTensorV2()
                                ], 
                            additional_targets = {'image':'image', 'mask': 'mask'}) 
    
    valid_transform =  A.Compose([  A.Resize(img_height, img_width),
                                    A.Normalize(mean=(0,0,0),std=(1,1,1)),  
                                    ToTensorV2()
                                ], 
                            additional_targets = {'image':'image', 'mask': 'mask'}) 
    

    return train_transform, valid_transform



def parse_json(file:str) -> typing.List[str]:
    with open(file, encoding = 'utf-8-sig') as f: 
        data = json.load(f)
    return data['DataSets'][0]['Images']


def draw_poly(json_file: str, out_dir:str):
    images = parse_json(json_file)
    for image in images: 
        image_name = image['ImageName']
        image_size = image['ImageSize']
        w, h = image_size.get('Width'), image_size.get('Height')

        out_file = os.path.join(out_dir, image_name)
        mask = np.zeros((w, h), dtype = np.uint8)
        segmentations = image['Annotations'][0]['Segmentation']
        if segmentations:
            for segment in segmentations: 
                string = segment['Selection'][9:-2].split(',')   
                arr = []
                for data in string: 
                    arr.append(data.split(' '))
                arr = np.array(arr, np.float32).astype(np.int32)
                cv2.fillPoly(mask, [arr], color = (255, 255, 255))
        cv2.imwrite(out_file, mask)
