import numpy as np
import torch
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, ConcatDataset

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


def build_transform(image_set, img_size):
    bbox_params = A.BboxParams(format="albumentations", label_fields=["labels"],
                                clip=True, min_visibility=0.3)

    normalize = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if image_set == "train":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0), p=0.5),
            A.Affine(scale=(0.8, 1.2), translate_percent=(0.0, 0.2), p=0.5),
            A.ColorJitter(brightness=(1 / 1.5, 1.5), contrast=0, saturation=(1 / 1.5, 1.5), hue=0, p=0.5),
            A.Resize(img_size, img_size),
            normalize,
            ToTensorV2(),
        ], bbox_params=bbox_params)

    return A.Compose([
        A.Resize(img_size, img_size),
        normalize,
        ToTensorV2(),
    ], bbox_params=bbox_params)


class VOCYearDataset:
    def __init__(self, root, year, image_set, img_size):
        self.dataset = torchvision.datasets.VOCDetection(
            root=root, year=year, image_set=image_set, download=True
        )
        self.transform = build_transform("train" if image_set == "trainval" else "val", img_size)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        w, h = img.size
        img = np.array(img)

        boxes = []
        labels = []
        for obj in target["annotation"]["object"]:
            bbox = obj["bndbox"]
            xmin, ymin = float(bbox["xmin"]) / w, float(bbox["ymin"]) / h
            xmax, ymax = float(bbox["xmax"]) / w, float(bbox["ymax"]) / h
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(VOC_CLASSES.index(obj["name"]))

        result = self.transform(image=img, bboxes=boxes, labels=labels)
        img = result["image"]

        return img, result["bboxes"], result["labels"]


def collate_fn(batch):
    images, boxes, labels = zip(*batch)
    images = torch.stack(images)
    return images, list(boxes), list(labels)


def get_dataloader(root="./data", image_set="train", img_size=448, batch_size=8, shuffle=True):
    if image_set == "train":
        # VOC07 trainval + VOC12 trainval
        voc07 = VOCYearDataset(root=root, year="2007", image_set="trainval", img_size=img_size)
        voc12 = VOCYearDataset(root=root, year="2012", image_set="trainval", img_size=img_size)
        dataset = ConcatDataset([voc07, voc12])
    else:
        # VOC07 test: publicly labeled
        dataset = VOCYearDataset(root=root, year="2007", image_set="test", img_size=img_size)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)