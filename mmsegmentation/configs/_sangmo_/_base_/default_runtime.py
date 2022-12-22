# yapf:disable
import wandb
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        dict(
            type='MMSegWandbHook', 
            init_kwargs = dict(
                project = "ImageSegmentation" 
            ),
            interval = 10, 
            log_checkpoint = True,
            log_checkpoint_metadata = True, 
            num_eval_images = 20
        )
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
