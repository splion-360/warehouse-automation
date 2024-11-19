import os
import torch 
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import argparse
from get_data import prepare_train_test_valid, get_data_loaders
from model import SegNet
import utils
import wandb


class Trainer:
    def __init__(self, args):
        self.start_epoch = 0
        self.best_mAP = 0
        
        args.device = torch.device('cuda:{}'.format(0) if args.device == 'cuda' else 'cpu')
        prepare_train_test_valid(args.data_dir, args.seed)
        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(args.data_dir, args.height, args.width, args.device, args.batch_size)
        self.model = SegNet(args.in_channels, args.out_channels)
        self.model.to(args.device)

        self.optim     = torch.optim.SGD(self.model.parameters(), lr = args.lr_init, momentum = args.momentum, weight_decay = args.weight_decay)
        self.criterion = nn.BCEWithLogitsLoss()

        self.iou_thresh = args.iou_thresh

        if args.resume: 
            self.load_model(args.weights)
        
        self.scheduler = utils.CosineDecayLR(self.optim,  
                                       T_max = args.epochs * len(self.train_loader),
                                        lr_init = args.lr_init,
                                        lr_min = args.lr_end,
                                        warmup = args.burn_in * len(self.train_loader))
        

        if args.wandb:
            ## Wandb setup for logging purposes (only on the master node)
            wandb.login()
            # Wandb initialize
            self.run = wandb.init(project="peer-robotics", ## Name of the project 
                            config = {  "dataset":"Pallet (Floor Segmentation)",
                                        "learning_rate": args.lr_init,
                                        "epochs": args.epochs,
                                        "batch size":args.batch_size,
                                        "architecture": "segnet"
                                        },
                                        name="Training Binary Segmentation Model" # Name of the session
                    )

        

    def load_model(self, weight_path):
        last_weight = os.path.join(os.path.split(weight_path), "last.pt")
        chkpt = torch.load(last_weight, map_location=self.device)
        self.model.load_state_dict(chkpt['model'])
        self.start_epoch = chkpt['epoch'] + 1
        
        if chkpt['optim'] is not None:
            self.optim.load_state_dict(chkpt['optim'])
            self.best_mAP = chkpt['best_mAP']
        del chkpt

    def save_model(self, epoch, mAP, weight_path):
        if mAP > self.best_mAP:
            self.best_mAP = mAP
     
        best_model_file = os.path.join(weight_path, 'best.pt')
        last_model_file = os.path.join(weight_path, 'last.pt')
        state_dict  ={  'epoch' : epoch, 
                        'best_mAP' : self.best_mAP,
                        'model' : self.model.state_dict(),
                        'optim' : self.optim.state_dict()
                        }
        if self.best_mAP == mAP: 
            torch.save(state_dict, best_model_file)
    
        if epoch > 0 and epoch % 5 == 0:
            torch.save(state_dict, last_model_file)
        del state_dict
        
    def train(self): 
        print(f"Train data length : {len(self.train_loader)}")

        for epoch in range(self.start_epoch, args.epochs):
            self.model.train()

            for i, (img, mask) in enumerate(self.train_loader):
        
                img, mask = img.to(args.device), mask.to(args.device)
                pred = self.model(img)
                self.optim.zero_grad()
                
                loss = self.criterion(pred, mask)
                loss.backward()
                self.optim.step()
                if args.use_scheduler: 
                    self.scheduler.step(len(self.train_loader) * epoch + i)

                if i % 5 == 0:
                    if args.wandb: 
                        wandb.log(data = {'train loss' : loss.item()})
                    
                    s = ('Epoch:[ %d | %d ]    Batch:[ %d | %d ]   loss: %.3f    '
                         'lr: %g') % (epoch, args.epochs - 1, i, len(self.train_loader) - 1, loss.item(),
                                      self.optim.param_groups[0]['lr'])
                    print(s)

                
            ### Evaluate performance on the valid data 
            print("=" * 20 + "Validation" + "=" * 20)
            metrics = utils.Metrics(self.model, self.iou_thresh, args.device)
            with torch.no_grad():
                val_loss, acc, dice, iou, mAP = metrics(self.val_loader, self.criterion)
                s = ('val loss : %.3f, Accuracy : %.3f  Dice : %.3f  IoU : %.3f mAP@[%.2f | %.2f | %.2f] : %.3f') % (val_loss, acc.item(), dice.item(), iou.item(), self.iou_thresh[0],
                                                                                                 self.iou_thresh[2], self.iou_thresh[1], mAP.item())
                print(s)
                if args.wandb: 
                    wandb.log(data = {'val loss': val_loss})
                    wandb.log(data = {'acc' : acc, 'dice' : dice, 'mAP': mAP})

            self.save_model(epoch, mAP, args.weights)
            print('best mAP : %g' % self.best_mAP)  
        
        if args.wandb: 
            self.run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type = int, default = 200, help = "epochs to train the model")
    parser.add_argument('--weights', type = str, default = "./weights", help = "weights file path")
    parser.add_argument('--data-dir', type = str, default = "./data", help = "data file path")
    parser.add_argument('--in-channels', type = int, default = 3, help = "input channels")
    parser.add_argument('--out-channels', type = int, default = 1, help = "output channels")
    parser.add_argument('--device', type = str, default = "cuda", help = "device to use")
    parser.add_argument('--height', type = int, default = 416, help = "height of the image")
    parser.add_argument('--width', type = int, default = 416, help = "width of the image")
    parser.add_argument('--batch-size', type = int, default = 8, help = "batch size")
    parser.add_argument('--lr-init', type = float, default = 0.001, help = "initial learning rate")
    parser.add_argument('--lr-end', type = float, default = 0.00001, help = "final learning rate")
    parser.add_argument('--weight-decay', type = float, default = 0.0005, help = "weight decay")
    parser.add_argument('--momentum', type = float, default = 0.9, help = "momentum")
    parser.add_argument('--burn-in', type = int, default = 2, help = "warmup epochs")
    parser.add_argument('--wandb', action = 'store_true', default = False, help = "wandb for logging")
    parser.add_argument('--resume', action = 'store_true', default = False, help = "resume saved checkpoint")
    parser.add_argument('--seed', type = int, default = 123456789, help = "random seed for result reproduction")
    parser.add_argument('--iou_thresh', nargs = "*", default = [0.5, 0.95, 0.05], help = "IoU threshold range (start, stop, step)")
    parser.add_argument('--use-scheduler', action = 'store_true', default = False, help = "use scheduler")
    args = parser.parse_args()

    Trainer(args).train()