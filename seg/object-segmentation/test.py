from get_data import get_data_loaders
import argparse
import torch
import torch.nn as nn
import torchvision
import model 
import os 
import utils


class Tester: 
    def __init__(self, args): 
        self.args = args
        args.device = torch.device('cuda:{}'.format(0) if args.device == 'cuda' else 'cpu')
        _, _ , self.test_loader = get_data_loaders(args.data_dir, args.height, args.width, args.device) 
        self.model = model.SegNet(args.in_channels, args.out_channels).to(args.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.iou_thresh = args.iou_thresh

        self.load_model(args.weights)
        self.evaluator = utils.Metrics(self.model, self.iou_thresh, args.device)


    def load_model(self, weight_path):
        print('Loading model weights from {}'.format(weight_path))
        last_weight = os.path.join(weight_path, "best.pt")
        chkpt = torch.load(last_weight, map_location = self.args.device)
        self.model.load_state_dict(chkpt['model'])
        del chkpt
    

    def test(self):

        print("=" * 20 + "Testing" + "=" * 20)    
        self.model.eval()

        with torch.no_grad():
            val_loss, acc, dice, iou, mAP = self.evaluator(self.test_loader, self.criterion)
            s = ('test loss : %.3f, Accuracy : %.3f  Dice : %.3f  IoU : %.3f mAP@[%.2f | %.2f | %.2f] : %.3f') % (val_loss, acc.item(), dice.item(), iou.item(), self.iou_thresh[0],
                                                                                                self.iou_thresh[2], self.iou_thresh[1], mAP.item())
            print(s)

        root = self.args.out_dir
        img_path, gt_path, pred_path = os.path.join(root, 'imgs'), os.path.join(root, 'mask', 'gt'), os.path.join(root, 'mask', 'pred')
        
        with torch.no_grad():
            for i, (img, mask) in enumerate(self.test_loader):
                img, mask = img.to(self.args.device), mask.to(self.args.device)
                pred = torch.sigmoid(self.model(img)) 
                pred_mask = (pred > 0.5).float()
                torchvision.utils.save_image(img, os.path.join(img_path, str(i) + '.jpg'))
                torchvision.utils.save_image(mask, os.path.join(gt_path, str(i) + '.jpg'))
                torchvision.utils.save_image(pred_mask, os.path.join(pred_path, str(i) + '.jpg'))

            print('Saved images at {}'.format(root))

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type = str, default = "./weights", help = "weights file path")
    parser.add_argument('--data-dir', type = str, default = "./data", help = "data file path")
    parser.add_argument('--in-channels', type = int, default = 3, help = "input channels")
    parser.add_argument('--out-channels', type = int, default = 1, help = "output channels")
    parser.add_argument('--device', type = str, default = "cuda", help = "device to use")
    parser.add_argument('--height', type = int, default = 416, help = "height of the image")
    parser.add_argument('--width', type = int, default = 416, help = "width of the image")
    parser.add_argument('--iou_thresh', nargs = "*", default = [0.5, 0.95, 0.05], help = "IoU threshold range (start, stop, step)")
    parser.add_argument('--out-dir', type = str, default = "./result", help = "result directory")
    args = parser.parse_args()

    Tester(args).test()