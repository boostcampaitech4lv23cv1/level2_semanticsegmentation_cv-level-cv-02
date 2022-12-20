
import torch
import torch.nn as nn 
import torchvision
import torch.optim as optim
import os 
import numpy as np
from torch.utils.data import DataLoader
import argparse
import random
from pathlib import Path
from importlib import import_module
import glob
import re
import math
from tqdm import tqdm
import json

from base import CATEGORIES, DATASET_PATH, NUM_CLASSES
from dataset import CustomDataset , collate_fn
from augmentation import train_transform, val_transform, test_transform
from utils import add_hist, label_accuracy_score, CosineAnnealingWarmUpRestarts

## Fix seed
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

# train.json / validation.json / test.json 디렉토리 설정
train_path = DATASET_PATH + '/train.json'
val_path = DATASET_PATH + '/val.json'
test_path = DATASET_PATH + '/test.json'

#device 관련 설정
device = "cuda" if torch.cuda.is_available() else "cpu"


def train(model_dir, args):
    print(f'Start training..')
    print("Fix seed")
    seed_everything(args.seed)
    
    best_loss = 9999999
    
    save_dir = increment_path(os.path.join(model_dir, args.name))
    os.makedirs(save_dir, exist_ok = True)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    print("Your model will be saved at:", save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)
    
    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    #Dataset prepare
    train_dataset = CustomDataset(data_dir=train_path, mode='train', transform=train_transform)
    val_dataset = CustomDataset(data_dir=val_path, mode='val', transform=val_transform)

    # DataLoader prepare
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
        )
    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
        )
    
    #Model prepare
    model_module = getattr(import_module("model"), args.model)  # default: Baseline Model, FCN_ResNet50
    model = model_module(num_classes = NUM_CLASSES).to(device) 
    
    if args.load_from:
        print("Load from checkpoint")
        model.load_state_dict(torch.load(f"/opt/ml/input/code/{args.load_from}", map_location = device))
        print("Load complete")
    
    # Loss, Optimizer prepare
    criterion = nn.CrossEntropyLoss()
    
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: Adam
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-6
    ) 
    
    #(TODO) scheduler는 알아서 지정
    
    print("Scheduler 사용 :", args.scheduler)
    
    if args.scheduler == "false":
        print("scheduler를 선택하지 않으셨습니다.")
    
    elif args.scheduler == "multi_sched":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs // 2], gamma=0.1)
    
    elif args.scheduler == "cosan_sched":
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0 = 10, T_mult= 2 , eta_max =0.05, T_up = 2, gamma = 0.5)

    else: 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, milestones=[args.epochs // 2], gamma=0.1)


    #Early stopping을 위한 patience 지정
    patience_check = 0
    
    for epoch in range(args.epochs):
        model.train()

        hist = np.zeros((NUM_CLASSES, NUM_CLASSES))
        train_num_batches = math.ceil(len(train_dataset) / args.batch_size)
        with tqdm(total = train_num_batches) as pbar:
            for step, (images, masks, _) in enumerate(train_loader):
                if args.val_debug== "true" : 
                    print("You set to val_debug mode as True. This is to check only validation logic, ignoring all train loop.")
                    break 
                
                pbar.set_description(f"[Epoch {epoch+1}/{args.epochs}], Step [{step+1}/{len(train_loader)}]")
                images = torch.stack(images)       
                masks = torch.stack(masks).long() 
                
                # gpu 연산을 위해 device 할당
                images, masks = images.to(device), masks.to(device)
                            
                # inference
                outputs = model(images)['out']
                
                # loss 계산 (cross entropy loss)
                loss = criterion(outputs, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                masks = masks.detach().cpu().numpy()
                
                hist = add_hist(hist, masks, outputs, n_class= NUM_CLASSES)
                acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
                
                if args.scheduler == "false":
                    current_lr = get_lr(optimizer)
                else:
                    current_lr = scheduler.get_lr()[0]
                postfix_dict = dict(
                    Acc = acc,
                    mIoU = round(mIoU,4),
                    Loss = round(loss.item(),4),
                    lr = current_lr
                )
                pbar.update(1)
                pbar.set_postfix(postfix_dict)
                                
            # validation 주기에 따른 loss 출력 및 best model 저장
            if (epoch + 1) % args.val_every == 0:
                avrg_loss = validation(epoch + 1, model, val_loader, criterion, device)
                if avrg_loss < best_loss:
                    print(f"Best performance at epoch: {epoch + 1}")
                    print(f"Save model in {save_dir}")
                    print(f"Best loss: {best_loss} -> {avrg_loss}")
                    best_loss = avrg_loss
                    torch.save(model.state_dict(), f"{save_dir}/best.pth")
                    patience_check = 0
                
                else:
                    patience_check +=1
                    
                    if patience_check == args.patience_limit : 
                        print("EARLY STOPPING")
                        break
                                    
            if (epoch +1) % args.save_model_interval == 0:
                print(f"Saving latest model at epoch {epoch+1}")
                torch.save(model.state_dict(), f"{save_dir}/latest.pth")
        
        if args.scheduler != "false":
            scheduler.step()
            
            
def validation(epoch, model, data_loader, criterion, device):
    print(f'Start validation #{epoch}')
    model.eval()

    with torch.no_grad():
        total_loss = 0
        cnt = 0
        
        hist = np.zeros((NUM_CLASSES, NUM_CLASSES))
        val_num_batches = len(data_loader)
        with tqdm(total = val_num_batches) as pbar:
            for step, (images, masks, _) in enumerate(data_loader):
                pbar.set_description(f"[Epoch {epoch+1} validation] , Step [{step+1} / {len(data_loader)}]")
                pbar.set_description("")
                
                
                images = torch.stack(images)       
                masks = torch.stack(masks).long()  

                images, masks = images.to(device), masks.to(device)            
                
                # device 할당
                model = model.to(device)
                
                outputs = model(images)['out']
                loss = criterion(outputs, masks)
                total_loss += loss
                cnt += 1
                
                outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                masks = masks.detach().cpu().numpy()
                
                hist = add_hist(hist, masks, outputs, n_class=NUM_CLASSES)
            
                acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
                postfix_dict = dict(
                        Acc = acc,
                        mIoU = round(mIoU,4),
                        Loss = round(loss.item(),4)
                    )
                
                pbar.update(1)
                pbar.set_postfix(postfix_dict)
                        
            IoU_by_class = [{classes : round(IoU,4)} for IoU, classes in zip(IoU , CATEGORIES)]
            
            avrg_loss = total_loss / cnt
            print(f'Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, \
                    mIoU: {round(mIoU, 4)}')
            print(f'IoU by class : {IoU_by_class}')
    return avrg_loss

# 모델 저장 함수 정의
val_every = 1


if __name__ == "__main__":
    # Data and model checkpoints directories
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train (default: 30)')
    parser.add_argument('--model', type=str, default='FCN_ResNet50', help='model type (default: FCN_ResNet50)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--val_every', type=int, default=1, help='how often will you log validation(default: 1)')
    parser.add_argument('--val_debug', type=str, default="false", help='If you want to skip train_loop, set to true (default: false)')
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size for training (default: 64)')
    parser.add_argument('--patience_limit', type=int, default=10, help='How many epochs for early stopping')
    parser.add_argument('--save_model_interval', type=int, default=10, help='Epochs for saving latest model')
    parser.add_argument('--scheduler', type=str, default="false", help='scheduler')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--load_from', type=str, default="", help='Model checkpoint')
    
    
    
    # 여기부턴 이후 추가될 수 있는 argparse
    parser.add_argument('--dataset', type=str, default='CustomDataset', help='dataset augmentation type (default: CustomDataset)')
    parser.add_argument('--augmentation', type=str, default='Baseline Augmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model_ckpt'))
    
    
    args = parser.parse_args()
    print(args)
    
    model_dir = args.model_dir
    train(model_dir, args)