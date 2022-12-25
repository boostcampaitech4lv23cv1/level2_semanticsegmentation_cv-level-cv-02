# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import matplotlib.pyplot as plt
import mmcv
import numpy as np
from matplotlib.ticker import MultipleLocator
from mmcv import Config, DictAction
import cv2
from mmseg.datasets import build_dataset
from tqdm import tqdm

from matplotlib.patches import Patch

import webcolors
import wandb
import warnings 
warnings.filterwarnings("ignore")

PALETTE = [[0, 0, 0], [192, 0, 128], [0, 128, 192], [0, 128, 64], [128, 0, 0],
           [64, 0, 128], [64, 0, 192], [192, 128, 64], [192, 192, 128],
           [64, 64, 128], [128, 0, 192]]

NUM_CLASSES = 11
CATEGORIES = ['Backgroud','General trash', 'Paper', 'Paper pack', 'Metal',
 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'] 

CLASS_MAPPINGS = dict(enumerate(CATEGORIES))

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate confusion matrix from segmentation results')
    parser.add_argument('--config', help='test config file path', 
                        default = "/opt/ml/mmsegmentation/configs/_sangmo_/swin/test_swin_large_example.py"
                        )
    parser.add_argument(
        '--prediction_path', help='prediction path where test .pkl result',
        default = "/opt/ml/mmsegmentation/result/pred_result.pkl"
        )
    
    parser.add_argument(
        "--image_numbers", help='how many images to loss',
        type = int,
        default = 30
    )
    
    parser.add_argument(
        "--reverse", help='if specified, plot top best samples',
        type = str,
        default = "false"
    )
    
    args = parser.parse_args()
    return args

def calculate_iou(gt_mask, pred_mask, class_type=1):
    pred_mask = (pred_mask == class_type) * 1
    gt_mask = (gt_mask == class_type) * 1
    overlap = pred_mask * gt_mask  # Logical AND
    union = (pred_mask + gt_mask) >0  # Logical OR
    iou = overlap.sum() / float(union.sum())
    
    if union.sum() == 0 : iou = np.nan 
    
    return iou

def get_total_predictions(dataset, results):
    informations = []
    preds = [0 for _ in range(NUM_CLASSES)]
    for idx, per_img_res in tqdm(enumerate(results)):
        res_segm = per_img_res
        gt_segm = dataset.get_gt_seg_map_by_idx(idx).astype(int)
        ious = np.array([calculate_iou(gt_segm, res_segm, obj_type) for obj_type in range(NUM_CLASSES)])
        miou = np.nanmean(ious)
        informations.append((idx, miou, ious))
    return informations 

def create_trash_label_colormap():
    """Creates a label colormap used in Trash segmentation.
    Returns:
        A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((11, 3), dtype=np.uint8)
    for inex, palette_val in enumerate(PALETTE):
        colormap[inex] = palette_val
    
    return colormap

def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
        label: A 2D array with integer type, storing the segmentation label.

    Returns:
        result: A 2D array with floating type. The element of the array
                is the color indexed by the corresponding element in the input label
                to the trash color map.

    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
              map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_trash_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]

def plot_examples(files, image_collections, num_images = None):
    """Visualization of images and masks according to batch size
    Args:
        mode: train/val/test (str)
        batch_id : 0 (int) 
        num_examples : 1 ~ batch_size(e.g. 8) (int)
        dataloaer : data_loader (dataloader) 
    Returns:
        None
    """
    # variable for legend
    category_and_rgb = [[category, palette] for(category, palette) in zip(CATEGORIES, PALETTE)]
    legend_elements = [Patch(facecolor=webcolors.rgb_to_hex(rgb), 
                             edgecolor=webcolors.rgb_to_hex(rgb), 
                             label=category) for category, rgb in category_and_rgb]
    
    if num_images is None:
        num_images = len(image_collections)
    
    # test / validation set에 대한 시각화 
    
    fig, ax = plt.subplots(nrows=num_images, ncols=3, figsize=(12, 4*num_images), constrained_layout=True)
    
    if len(ax.shape) == 1 : ax = ax.reshape(1,-1)
    
    for row_num, (idx_num, mask_gt, mask_pred) in enumerate(image_collections):
        if row_num >= num_images: break
        # Original Image
        image = cv2.imread(files[idx_num])[:,:,::-1]
        ax[row_num][0].imshow(image)
        ax[row_num][0].set_title(f"Orignal Image : { files[idx_num].split('/')[-1] } ")
        # Groud Truth
        ax[row_num][1].imshow(label_to_color_image(mask_gt))
        ax[row_num][1].set_title(f"Groud Truth : {files[idx_num].split('/')[-1]}")
        # Pred Mask
        ax[row_num][2].imshow(label_to_color_image(mask_pred))
        ax[row_num][2].set_title(f"Pred Mask : {files[idx_num].split('/')[-1]}")
        ax[row_num][2].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    return fig, ax



def main(args):
    #(TODO) 향후 argparse로 대체     
    cfg = Config.fromfile(args.config) #FIXME
    results = mmcv.load(args.prediction_path) #FIXME
    image_numbers = args.image_numbers
    reverse = args.reverse
    
    dataset = build_dataset(cfg.data.test)
    ignore_index = dataset.ignore_index
    n = len(dataset.CLASSES)
    VAL_PATH = cfg.data.test.img_dir
    FILES = sorted([os.path.join(VAL_PATH, fname) for fname in os.listdir(VAL_PATH)])
    
    print("wandb를 켜는 중입니다.")
    
    wandb.init(
        project = "Seg_vizualization"
    )
    columns = ["Image", "Comparisons" , "mIoU", "Classwise IoU"]
    
    wandb_result_table = wandb.Table(columns = columns)
    
    print("전체 예측 결과를 계산 중입니다..")

    informations = get_total_predictions(dataset, results)        
    mious = np.array([ex[1] for ex in informations])
    total_miou = np.nanmean(mious)
    

    print("현재 CV와 어느정도의 차이가 있어 확인 중이나, 파일 자체는 잘 돌아갑니다. 이 점 염두해주시기 바랍니다.")
    print("Total Val mIoU:", total_miou)

    if reverse == "true":
        print(f"Reverse 옵션을 지정하셨습니다. Best K = {image_numbers} 개의 mIoU를 출력합니다.")
        chosen = sorted(informations, key = lambda x: x[1])[-image_numbers:]
    else:
        print(f"Worst K = {image_numbers} 개의 mIoU를 출력합니다.")
        chosen = sorted(informations, key = lambda x: x[1])[:image_numbers]

    print("기록중입니다..")
    for img_idx, miou, sample_iou in tqdm(chosen):
        gt_segm = dataset.get_gt_seg_map_by_idx(img_idx).astype(int)
        res_segm = results[img_idx]
        image = cv2.imread(FILES[img_idx])[:,:,::-1]
        
        image_collections = [  [img_idx, gt_segm, res_segm] ]
        
        fig, ax = plot_examples(FILES, image_collections)
        
        mask_image = wandb.Image(image,
                                 masks = {
                                     "predictions" : {
                                         "mask_data" : res_segm,
                                         "class_labels" : CLASS_MAPPINGS
                                         },
                                     "ground_truth" : {
                                         "mask_data" : gt_segm,
                                         "class_labels" : CLASS_MAPPINGS 
                                     }
                                 }
                                 )
        comparison = wandb.Image(fig)
        
        sample_iou_txt = str({cat: round(iou,2) for cat, iou in zip(CATEGORIES, sample_iou) 
                              if iou>=0})
        
        wandb_result_table.add_data(
            mask_image,
            comparison,
            miou,
            sample_iou_txt
        )
    wandb.log({f"TOP {image_numbers} sample"  : wandb_result_table})
    wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    print("args:", args)
    main(args)