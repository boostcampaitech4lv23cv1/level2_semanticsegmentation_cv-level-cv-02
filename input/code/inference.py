import torch 
import numpy as np 
from tqdm import tqdm
import albumentations as A
from base import CATEGORIES, DATASET_PATH, NUM_CLASSES
from dataset import CustomDataset , collate_fn
from augmentation import train_transform, val_transform, test_transform
from torch.utils.data import DataLoader
import random
from importlib import import_module
import argparse
import os
import pandas as pd
import math

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
            num_classes=num_classes
    )

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

@torch.no_grad()
def inference(model_dir, args):

    # Dataset 관련 지정    
    test_path = DATASET_PATH + '/test.json'
    test_dataset = CustomDataset(data_dir=test_path, mode='test', transform=test_transform)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    #Model 관련 지정
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = load_model(model_dir, NUM_CLASSES, device).to(device)
    model.eval()
    size = 256
    transform = A.Compose([A.Resize(size, size)])
    print('Start prediction.')
    

    
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    with torch.no_grad():
        test_num_batches = math.ceil(len(test_dataset) / args.batch_size)
        with tqdm(total = test_num_batches) as pbar:
            for step, (imgs, image_infos) in enumerate(test_loader):
                pbar.set_description(f"Inference")
                # inference (512 x 512)
                outs = model(torch.stack(imgs).to(device))['out']
                oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
                
                # resize (256 x 256)
                temp_mask = []
                for img, mask in zip(np.stack(imgs), oms):
                    transformed = transform(image=img, mask=mask)
                    mask = transformed['mask']
                    temp_mask.append(mask)
                    
                oms = np.array(temp_mask)
                
                oms = oms.reshape([oms.shape[0], size*size]).astype(int)
                preds_array = np.vstack((preds_array, oms))
                
                file_name_list.append([i['file_name'] for i in image_infos])
                pbar.update(1)
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    
    return file_names, preds_array


#(TODO) ensemble 시 바뀔 수 있는 코드

def make_submission(model_dir, args):
    # sample_submisson.csv 열기
    submission = pd.read_csv('./submission/sample_submission.csv', index_col=None)

    # test set에 대한 prediction
    file_names, preds = inference(model_dir, args)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                    ignore_index=True)

    # submission.csv로 저장
    submission.to_csv("./submission/fcn_resnet50_best_model(pretrained).csv", index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size for validing (default: 4)')    
    parser.add_argument('--model', type=str, default='FCN_ResNet50', help='model type (default: Baseline model, FCN_ResNet50)')

    # Container environment
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model_ckpt/exp'))

    args = parser.parse_args()
    print(args)

    model_dir = args.model_dir

    #Change if you want to do kfold, or just use inference
    make_submission(model_dir, args)
    