# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=40)
checkpoint_config = dict(
    by_epoch = True, interval = 50, max_keep_ckpts = 3)
#classwise 별로 출력
evaluation = dict(metric='mIoU', pre_eval=True, 
                  classwise = True, save_best='mIoU')
work_dir = './work_dirs/baseline'
gpu_ids = [0]
auto_resume = False