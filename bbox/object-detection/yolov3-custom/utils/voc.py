import sys
sys.path.append("..")
import xml.etree.ElementTree as ET
import config.yolov3_config_voc as cfg
import os
from tqdm import tqdm

def parse_voc_annotation(data_path, anno_path, use_difficult_bbox=False):
    """
    pascal voc annotation,[image_global_path xmin,ymin,xmax,ymax,cls_id]
    :param data_path: D:\doc\data\VOC\VOCtrainval-2007\VOCdevkit\VOC2007
    :param file_type: 'trainval''train''val'
    :param anno_path: 
    :param use_difficult_bbox:difficult==1 bbox
    """
    classes = cfg.DATA["CLASSES"]
    with open(data_path, 'r') as file: 
        images = file.readlines()
    
    n = 0
    ROOT = os.path.dirname(data_path)
    with open(anno_path, 'a') as f:
        for image_id in tqdm(images):
            image_name = image_id.rstrip().split('/')[-1]
            image_path = os.path.join(ROOT, 'imgs', image_name) 


            annotation = image_path
            label_path = os.path.join(ROOT, 'labels', image_name[:-4] + '.xml')
            root = ET.parse(label_path).getroot()
            objects = root.findall('object')
            root.find("filename").text = image_name[:-4] + '.xml'

            if root.find("path") is not None:  
                root.find("path").text = image_path

            if objects: 
                n += 1
                
                for obj in objects:
                    bbox = obj.find('bndbox')
                    
                    obj.find('difficult').text = "0"
                    obj.find('truncated').text = "0"
                    obj.find('pose').text = "Unspecified"          
                    try:
                        class_id = classes.index(obj.find("name").text.lower().strip())
                    except ValueError as e: 
                        print("Pallet Facade class replaced as pallet !!!")
                        class_id = 0
                        obj.find("name").text = classes[class_id]

                    xmin = bbox.find('xmin').text.strip().split('.')[0]
                    ymin = bbox.find('ymin').text.strip().split('.')[0]
                    xmax = bbox.find('xmax').text.strip().split('.')[0]
                    ymax = bbox.find('ymax').text.strip().split('.')[0]
                    annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_id)])
                    bbox.find('xmin').text = xmin 
                    bbox.find('ymin').text = ymin
                    bbox.find('xmax').text = xmax
                    bbox.find('ymax').text = ymax
                annotation += '\n'
                # print(annotation)
                f.write(annotation)
            tree = ET.ElementTree(root)
            tree.write(label_path)
    return n


if __name__ =="__main__":
    
    train_data_path = os.path.join(cfg.DATA_PATH, '..','data', 'train.txt')
    train_annotation_path = os.path.join(cfg.DATA_PATH, '..', 'train_annotation.txt')
    if os.path.exists(train_annotation_path):
        os.remove(train_annotation_path)

    valid_data_path = os.path.join(cfg.DATA_PATH, '..','data', 'val.txt')
    valid_annotation_path = os.path.join(cfg.DATA_PATH, '..', 'val_annotation.txt')
    if os.path.exists(valid_annotation_path):
        os.remove(valid_annotation_path)

    test_data_path = os.path.join(cfg.DATA_PATH, '..', 'data', 'test.txt')
    test_annotation_path = os.path.join(cfg.DATA_PATH, '..', 'test_annotation.txt')
    if os.path.exists(test_annotation_path):
        os.remove(test_annotation_path)

    len_train = parse_voc_annotation(train_data_path, train_annotation_path, use_difficult_bbox=False)
    len_valid = parse_voc_annotation(valid_data_path, valid_annotation_path, use_difficult_bbox=False)
    len_test = parse_voc_annotation(test_data_path, test_annotation_path, use_difficult_bbox=False)

    print("The number of images for train, valid and test are :train : {} | valid: {} | test : {}".format(len_train, len_valid, len_test))
