"""
File containing the torch dataloaders for training detection and segmentation models

"""
import typing 
import torch.utils.data as dataset
import numpy as np
import os 
from PIL import Image
import utils


class PalletDataset(dataset.Dataset):
    '''
    Dataloader for the segmentation model
    '''
   
    def __init__(self, root : str, type : str = 'train', transform = None): 
        self.root = root
        self.img_dir = os.path.join(root, type + '.txt')
        self.transform  = transform
        
        with open(self.img_dir, 'r') as file: 
            self.images = file.readlines()

    def __getitem__(self, idx : int) -> typing.Tuple[np.ndarray, np.ndarray]:
        img_path = self.images[idx].rstrip()
        img_name = img_path.split('/')[-1]
        label_path = os.path.join(self.root, 'mask', img_name)
        
        img   = np.array(Image.open(img_path))
        mask = np.array(Image.open(label_path), dtype = np.float32)
    
        mask /= (mask.max() + 1e-07)  ## Prevent numerical underflow
        
        if self.transform is not None: 
            transform = self.transform(image = img, mask = mask)
            img   = transform["image"]
            mask = transform["mask"][None,:,:]

        return img, mask 

    def __len__(self) -> int: 
        return len(self.images)
    

def prepare_train_test_valid(data_dir : str, seed : int , split : tuple = (0.75, 0.20)) -> None:
    ROOT = data_dir
    train_split, val_split = split 
    np.random.seed(seed)

    ## Contains the file paths to the images
    train_file, val_file, test_file = os.path.join(ROOT, 'train.txt'), os.path.join(ROOT, 'val.txt'), os.path.join(ROOT, 'test.txt')
    
    if os.path.exists(train_file):
        os.remove(train_file)
    if os.path.exists(val_file):
        os.remove(val_file)
    if os.path.exists(test_file):
        os.remove(test_file)  

    images = os.listdir(os.path.join(ROOT, 'imgs'))
    N = len(images)
    indices = [i for i in range(N)]
    np.random.shuffle(indices)

    train_i, val_i = int(train_split * N), int(val_split * N)
    train_indices, val_indices, test_indices = indices[:train_i], indices[train_i: train_i + val_i], indices[train_i + val_i:]

    ## Write to files
    with open(train_file, 'a') as train:
        for i in train_indices: 
            filename = os.path.join(ROOT, 'imgs', str(i) + '.jpg')
            filename += '\n'
            train.write(filename) 

    with open(val_file, 'a') as val:
        for i in val_indices: 
            filename = os.path.join(ROOT, 'imgs', str(i) + '.jpg')
            filename += '\n'
            val.write(filename) 

    with open(test_file, 'a') as test:
        for i in test_indices: 
            filename = os.path.join(ROOT, 'imgs', str(i) + '.jpg')
            filename += '\n'
            test.write(filename) 

    print("Texts generated at {}, {} and {}".format(train_file, val_file, test_file))


def get_data_loaders(data_dir : str, 
                     img_height: int = 320, 
                     img_width: int = 480, 
                     device : str = 'cuda',
                     batch_size : int = 2):

    train_transform, val_transform = utils.get_transforms(img_height, img_width, **{'flip' : 0.5})

    train_dataset = PalletDataset(data_dir, 'train',train_transform)
    val_dataset   = PalletDataset(data_dir, 'val',  val_transform)
    test_dataset  = PalletDataset(data_dir, 'test', val_transform)

    ## Prepare dataloaders
    pin_memory = device == 'cuda'
    train_loader = dataset.DataLoader(train_dataset, batch_size, shuffle = True, pin_memory = pin_memory)
    val_loader   = dataset.DataLoader(val_dataset, 1, shuffle = False, pin_memory = pin_memory)
    test_loader  = dataset.DataLoader(test_dataset, 1, shuffle = False, pin_memory = pin_memory)

    return train_loader, val_loader, test_loader





if __name__ == "__main__":
    data_dir = "./data"
    prepare_train_test_valid(data_dir, seed = 23)
    train, val, test = get_data_loaders(data_dir, 416, 416)
    breakpoint()
