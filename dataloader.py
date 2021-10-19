import cv2

from torch.utils.data import Dataset, DataLoader
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, ImageCompression,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomCrop, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout,
    ShiftScaleRotate, CenterCrop, Resize,
    ChannelShuffle, LongestMaxSize, HueSaturationValue, ISONoise
)

from albumentations.pytorch import ToTensorV2
import albumentations as A

def get_transforms(args, data):
    if data == 'train':
        return A.Compose([
            Resize(args.img_size[0], args.img_size[1]),
            Blur(),
            RandomBrightnessContrast(),
            HueSaturationValue(),
            CoarseDropout(max_holes=4, max_height=16, max_width=16, p=0.5),
            ShiftScaleRotate(rotate_limit=0, p=0.5),
            ImageCompression(quality_lower=40, quality_upper=80, p=0.5),
            GaussNoise(p=0.5),
            ToTensorV2(transpose_mask=False)
        ], p=1.)

    elif data == 'valid':
        return A.Compose([
            Resize(args.img_size[0], args.img_size[1]),
            ToTensorV2(transpose_mask=False)
        ], p=1.)

class MotionDataSet(Dataset):
    def __init__(self, data, transform=None, test=False):
        self.data = data  # dataframe
        self.test = test  # bool
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name_input = self.data.iloc[idx]['path']
        if self.test:  # test
            images = cv2.imread(file_name_input)
            if self.transform:
                transformed = self.transform(image=images)
                images = transformed["image"] / 255.
            return images

        else:  # train
            images = cv2.imread(file_name_input)
            targets = self.data.iloc[idx]['target']
            if self.transform:
                transformed = self.transform(image=images)
                images = transformed["image"] / 255.

            return images, targets