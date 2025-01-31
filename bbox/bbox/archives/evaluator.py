import config.yolov3_config_voc as cfg
import os
import shutil
from eval import voc_eval
from utils.datasets import *
from utils.gpu import *
import cv2
import numpy as np
from utils.data_augment import *
import torch
from utils.tools import *
from tqdm import tqdm
from utils.visualize import *
import torchvision

class Evaluator(object):
    def __init__(self, model, visiual=True,save_imgs=False):
        self.classes = cfg.DATA["CLASSES"]
        self.pred_result_path = os.path.join(cfg.PROJECT_PATH, 'results', 'voc')
        self.val_data_path = os.path.join(cfg.DATA_PATH, 'data', 'val.txt')
        self.conf_thresh = cfg.TEST["CONF_THRESH"]
        self.nms_thresh = cfg.TEST["NMS_THRESH"]
        self.val_shape =  cfg.TEST["TEST_IMG_SIZE"]
        self.ROOT = os.path.dirname(self.val_data_path)

        self.__visiual = visiual
        self.__visual_imgs = 0
        self.gan_model = None
        self.model = model
        self.device = next(model.parameters()).device
        self.__generator_transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                        torchvision.transforms.Resize((448,448)),
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]
                                                        )])
        self.save_imgs = save_imgs

    def APs_voc(self, multi_test=False, flip_test=False,direct_flag=False):
        with open(self.val_data_path, 'r') as file: 
            images = file.readlines()

        if os.path.exists(self.pred_result_path):
            shutil.rmtree(self.pred_result_path)
        os.makedirs(self.pred_result_path)

        for label in cfg.DATA['CLASSES']:
            filename = 'comp4_det_test_'+label+'.txt'
            os.system('touch results/voc/'+filename)

        for img_id in tqdm(images):
            img_name = img_id.rstrip().split('/')[-1]
            img_path = os.path.join(self.ROOT, 'imgs', img_name)
            img = cv2.imread(img_path)
            #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            bboxes_prd = self.get_bbox(img, multi_test, flip_test,direct_flag)
         
            if bboxes_prd.shape[0]!=0 and self.save_imgs: #and self.__visiual and self.__visual_imgs < 100:
                boxes = bboxes_prd[..., :4]
                class_inds = bboxes_prd[..., 5].astype(np.int32)
                scores = bboxes_prd[..., 4]
                #img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

                visualize_boxes(image=img, boxes=boxes, labels=class_inds, probs=scores, class_labels=self.classes)
                path = os.path.join(self.pred_result_path, "{}.jpg".format(self.__visual_imgs))
                cv2.imwrite(path, img)

                self.__visual_imgs += 1

            for bbox in bboxes_prd:
                coor = np.array(bbox[:4], dtype=np.int32)
                score = bbox[4]
                class_ind = int(bbox[5])

                class_name = self.classes[class_ind]
                score = '%.4f' % score
                xmin, ymin, xmax, ymax = map(str, coor)
                s = ' '.join([img_name[:-4], score, xmin, ymin, xmax, ymax]) + '\n'

                with open(os.path.join(self.pred_result_path, 'comp4_det_test_' + class_name + '.txt'), 'a') as f:
                    f.write(s)
            
        return self.__calc_APs()

    def get_bbox(self, img, multi_test=False, flip_test=False,direct_flag=False):
        if multi_test:
            test_input_sizes = range(320, 640, 96)
            bboxes_list = []
            for test_input_size in test_input_sizes:
                valid_scale =(0, np.inf)
                bboxes_list.append(self.__predict(img, test_input_size, valid_scale))
                if flip_test:
                    bboxes_flip = self.__predict(img[:, ::-1], test_input_size, valid_scale)
                    bboxes_flip[:, [0, 2]] = img.shape[1] - bboxes_flip[:, [2, 0]]
                    bboxes_list.append(bboxes_flip)
            bboxes = np.row_stack(bboxes_list)
        else:
            if not direct_flag:
                bboxes = self.__predict(img, self.val_shape, (0, np.inf))
            else:
                
                bboxes = self.__predict_direct(img, self.val_shape, (0, np.inf))

        bboxes = nms(bboxes, self.conf_thresh, self.nms_thresh)
        return bboxes

    def __predict(self, img, test_shape, valid_scale):
        org_img = np.copy(img)
        org_h, org_w, _ = org_img.shape

        img = self.__get_img_tensor(img, test_shape).to(self.device)
        self.model.eval()
        with torch.no_grad():
            _, p_d = self.model(img)
        pred_bbox = p_d.squeeze().cpu().numpy()
        bboxes = self.__convert_pred(pred_bbox, test_shape, (org_h, org_w), valid_scale)
        

        return bboxes
    def __predict_direct(self, img, test_shape, valid_scale):
        org_h, org_w,_  = img.shape
        img = self.__generator_transform(img).float().unsqueeze(0).cuda()
        self.model.eval()
        self.gan_model.eval()
        with torch.no_grad():
            out_img = self.gan_model(img)
            out_img = (out_img*0.5)+0.5
            resize=torchvision.transforms.Resize(( self.val_shape, self.val_shape))
            out_img = resize(out_img)
            _, p_d = self.model(out_img)
        pred_bbox = p_d.squeeze().cpu().numpy()
        bboxes = self.__convert_pred(pred_bbox, test_shape, ( self.val_shape, self.val_shape), valid_scale)
        return bboxes

    def __get_img_tensor(self, img, test_shape):
        img = Resize((test_shape, test_shape), correct_box=False)(img, None).transpose(2, 0, 1)
        return torch.from_numpy(img[np.newaxis, ...]).float()


    def __convert_pred(self, pred_bbox, test_input_size, org_img_shape, valid_scale):
        """
        """
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


    def __calc_APs(self, iou_thresh=0.5, use_07_metric=False):
        """
        :param iou_thresh:
        :param use_07_metric:
        :return:dict{cls:ap}
        """

        filename = os.path.join(self.pred_result_path, 'comp4_det_test_{:s}.txt')
        cachedir = os.path.join(self.pred_result_path, 'cache')
        annopath = os.path.join(self.ROOT, 'labels', '{:s}.xml')
        imagesetfile = self.val_data_path
        APs = {}
        for i, cls in enumerate(self.classes):
            R, P, AP = voc_eval.voc_eval(filename, annopath, imagesetfile, cls, cachedir, iou_thresh, use_07_metric)
            APs[cls] = AP
        if os.path.exists(cachedir):
            shutil.rmtree(cachedir)

        return APs
