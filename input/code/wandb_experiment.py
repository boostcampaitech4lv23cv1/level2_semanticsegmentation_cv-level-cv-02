import os 
import wandb 
from base import CATEGORIES
import matplotlib.pyplot as plt
import numpy as np 
import random
class ExperimentWandb:
    def __init__(self, project_name = "ImageSegmentation", run_name = "Wandb_EfficientB4"):
        self.project_name = project_name 
        self.run_name = run_name
        self.batch_size = 32
        self.lr = 1e-4 
        self.epochs = 10 
    
    def config(self, config_dict = None):
        wandb.init(
            project = self.project_name,
            config = dict(epochs = self.epochs,
            batch_size = self.batch_size,
            lr = self.lr) if config_dict is None else config_dict,
        )

        wandb.run_name = self.run_name

    def set_project_name(self, new_name):
        self.project_name = new_name 

    def set_run_name(self, run_name):
        self.run_name = run_name 
    
    def set_hyperparams(self, hyopt_configs):
        for config_name, config_value in hyopt_configs.items():
            setattr(self, config_name, config_value)

    def log(self, *args, **kwargs):
        wandb.log(*args, **kwargs)


    def check_all_iou(self, classwise_iou):
        '''
        Epoch당 classwise IoU를 기록합니다.
        '''
        for per_category in classwise_iou:
            for cat_name, class_iou in per_category.items():
                logs_to_write = {cat_name : round(float(class_iou), 4)}
                wandb.log(logs_to_write)

    # def single_plot_image(self, log_type):
    #     """
    #     log에 하고자 하는 클래스만 이용하면 충분!
    #     """
    #     fig, ax = plt.subplots(1,1, figsize = (12,12))
        
    #     #Acc_age_old, Acc_total 이런 느낌으로 2번째 것을 추출하면 충분
    #     image_type = list(log_type.keys())[0].split("_")[1].upper()
    #     title = f"{image_type} Classify"

    #     length = len(log_type)

    #     original_x_values = list(range(length))
    #     colorbars = ["skyblue", "violet", "purple", "salmon", "magenta", "green", "blue", "brown", "black"]
    #     clist = random.sample(colorbars, length)

    #     ax.set_xticks(original_x_values)

    #     x_values = list(log_type.keys())
    #     y_values = list(log_type.values())

    #     #모든 prediction은 0과 1사이
    #     ax.set_ylim(0,1)
    #     ax.set_xticklabels(x_values, fontsize = 12)

    #     #그래프 그리기
    #     ax.bar(x_values, y_values, color = clist)

    #     for x, y in zip(original_x_values, y_values):
    #         ax.text(x, y+ 0.05, str(round(y,2)), color=colorbars[x], fontweight='bold', ha = "center", fontsize = 15)

    #     #경계선 제거
    #     ax.spines.right.set_visible(False)
    #     ax.spines.top.set_visible(False)

    #     ax.set_title(title, fontsize = 20)
    #     plt.savefig(f"{image_type}.png",dpi=300)
    #     return fig, ax

    # def multiple_plot_image(self, log_types):
    #     """
    #     log에 하고자 하는 클래스만 이용하면 충분!
    #     """
    #     fig, ax = plt.subplots(1,3, figsize = (20,8))
        
    #     fig.suptitle("Multiple class accuracy", fontsize = 25)
        
    #     for idx, log_type in enumerate(log_types):
    #         #Acc_age_old, Acc_total 이런 느낌으로 2번째 것을 추출하면 충분
    #         image_type = list(log_type.keys())[0].split("_")[1].upper()
    #         title = f"{image_type} Classify"

    #         length = len(log_type)

    #         original_x_values = list(range(length))
    #         colorbars = ["skyblue", "violet", "purple", "salmon", "magenta", "green", "blue", "brown", "black"]
    #         clist = random.sample(colorbars, length)

    #         ax[idx].set_xticks(original_x_values)

    #         x_values = list(log_type.keys())
    #         y_values = list(log_type.values())

    #         #모든 prediction은 0과 1사이
    #         ax[idx].set_ylim(0,1)
    #         ax[idx].set_xticklabels(x_values, fontsize = 8)

    #         #그래프 그리기
    #         ax[idx].bar(x_values, y_values, color = clist)

    #         for x, y in zip(original_x_values, y_values):
    #             ax[idx].text(x, y+ 0.05, str(round(y,2)), color=colorbars[x], fontweight='bold', ha = "center", fontsize = 15)

    #         #경계선 제거
    #         ax[idx].spines.right.set_visible(False)
    #         ax[idx].spines.top.set_visible(False)

    #         ax[idx].set_title(title, fontsize = 15)
        
    #     plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.3)
    #     plt.savefig("Multiple.png",dpi=300)

    #     return fig, ax

    # def plot_best_accuracy(self, correct_cnt: np.array, total_cnt: np.array) -> None:
    #     """
    #     All accuracy
    #     """
        
    #     plt.rcParams['figure.dpi'] = 150

    #     mask_data, gender_data, age_data, total_data = self.check_all_accuracy(correct_cnt, total_cnt)

    #     fig, ax = plt.subplots(1,3, figsize = (20,8))

    #     fig.suptitle("Multiple class accuracy", fontsize = 25)

    #     log_types = [mask_data, gender_data, age_data]
    
    #     fig_3way , _ = self.multiple_plot_image(log_types)
    #     fig_total, _ = self.single_plot_image(total_data)

    #     wandb.log({"Fig_multi_class" :  fig_3way})
    #     wandb.log({"Fig_single" :  fig_total})



    def log_miss_label(self, miss_labels: list) -> None:
        '''
        Valid Dataset에서 잘못 라벨링 한 데이터의 이미지와 추론값을 표로 출력합니다.
        :miss_label - 잘못 라벨링 된 데이터의 정보를 담은 리스트, [(img, label, pred)] 형식으로 저장

        wandb table을 이용함
        '''
        missing_table = wandb.Table(columns = ["image", "label", "pred", "Age", "Gender", "Mask"])

        for image, label, pred in miss_labels:
            image = image.transpose(1,2,0)
            image = wandb.Image(image)

            mask_label , mask_pred = label//6  , pred //6
            gender_label , gender_pred = (label - 6 * mask_label) //3 ,  (pred - 6 * mask_pred) //3 
            age_label, age_pred = (label- 6 * mask_label) % 3   , (pred - 6 * mask_pred) % 3

            mask_content = f"Mask_GT : {mask_label}, Mask_pred : {mask_pred}"
            gender_content = f"gender_GT : {gender_label}, gender_pred : {gender_pred}"
            age_content = f"Age_GT : {age_label}, Age_pred : {age_pred}"

            missing_table.add_data(image, label, pred, age_content, gender_content, mask_content)

        wandb.log({"Miss Table": missing_table}, commit = False)
    
    def finish(self):
        wandb.finish()
  

wandb.login()