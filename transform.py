import albumentations as A
from albumentations.pytorch import ToTensorV2


class CustomTransform:
    def __init__(self, train=True):
        self.train = train

    def __call__(self, image, label):
        if self.train:
            transform = A.Compose([
                A.Resize(448, 448),
                A.ColorJitter(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5, rotate_limit=0),
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', min_area=1024, min_visibility=0.3))
        else:
            transform = A.Compose([
                A.Resize(448, 448),
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
                ToTensorV2(),
            ])
        return transform(image=image, bboxes=label)
