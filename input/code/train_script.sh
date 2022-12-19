#!/usr/bin/env bash

# model: 학습에 사용할 모델. model.py에 정의되어있으며, 베이스라인 코드는 FCN_ResNet50으로 구현했음
# optimizer : SGD, Adam, AdamW 등
# scheduler : None(default) | multi_sched(MultiStepLR) | cosan_sched(CosineAnnealing) | 그 외(CosineWarmupRestart)
# batch_size : 배치 사이즈. 기본은 4. 늘릴지 줄일지는 메모리 보고 결정하자.
# patience_limit : early stopping 용 epoch
# save_model_interval: latest.pth를 저장하고, wandb error case plot의 interval
# load_from : model을 불러올 체크포인틥니다.
# val_debug : Train_loop를 생략하고, val_loop로만 디버깅하고 싶을 시 사용합니다.

python train.py \
--epochs 30 \
--model "FCN_ResNet50" \
--optimizer "AdamW"     \
--scheduler "multi_sched" \
--batch_size 4 \
--patience_limit 10  \
--save_model_interval 10 \
#  --load_from model_ckpt/exp/best.pth  \
#  --val_debug true \