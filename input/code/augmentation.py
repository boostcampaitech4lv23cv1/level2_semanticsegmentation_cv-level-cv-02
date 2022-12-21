import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

imgnet_mean = (0.485, 0.456, 0.406)
imgnet_std = (0.229, 0.224, 0.225)

train_transform_pipeline = [
    A.OneOf([
        A.CropNonEmptyMaskIfExists(384, 384),
    ], p=0.5),
    A.Resize(384, 384, interpolation=cv2.INTER_AREA),
    A.OneOf([
        A.HorizontalFlip(),
        A.VerticalFlip(),
    ], p=0.67),
    A.OneOf([
        A.RandomRotate90(),
    ], p=0.67),
    A.RandomBrightnessContrast(p=0.5),
    A.OneOf([
        A.OpticalDistortion(),
        A.GridDistortion(),
    ], p=0.67),
    A.OneOf([
        A.Emboss(),
        A.Sharpen(),
    ], p=0.75),
    
    A.Normalize(mean=imgnet_mean, std=imgnet_std, max_pixel_value=1.0),
    ToTensorV2()
]

val_transform_pipeline =  [
    A.Resize(512, 512),
    A.Normalize(mean=imgnet_mean, std=imgnet_std, max_pixel_value=1.0),
    ToTensorV2()
]

train_transform = A.Compose(train_transform_pipeline)
val_transform = A.Compose(val_transform_pipeline)
test_transform = A.Compose(val_transform_pipeline)


# train_transform = A.Compose([
#                             ToTensorV2()
#                             ])

# val_transform = A.Compose([
#                           ToTensorV2()
#                           ])

# test_transform = A.Compose([
#                            ToTensorV2()
#                            ])