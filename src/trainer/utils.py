import random
from torch.utils.data import Dataset
import numpy as np
import cv2
import h5py
from PIL import Image
import pathlib
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(img_name, train=True, suffix='', crop=False, train_size=(720, 720)):
    dataset_path = pathlib.Path(__file__).parent.parent.parent / 'dataset'
    gt_path = dataset_path / suffix / f'{img_name}.h5'
    img_path = dataset_path / 'images' / f'{img_name}.jpg'
    print(gt_path)
    print(img_path)
    
    img = Image.open(img_path).convert("RGB")
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file["density"])

    img = np.array(img)
    cv2.circle(img, (1204, 633), 32, (0, 0, 0), thickness=-1)
    cv2.rectangle(img, (1100, 672), (1265, 685), (0, 0, 0), thickness=-1)
    cv2.rectangle(img, (1154, 685), (1265, 710), (0, 0, 0), thickness=-1)
    img = Image.fromarray(img)

    if train:
        if crop:
            crop_size = train_size
            dx = int(random.randint(0, 1) * (img.size[0] * 1. - crop_size[0]))
            dy = int(random.randint(0, 1) * (img.size[1] * 1. - crop_size[1]))

            img = img.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy)).copy()
            target = target[dy:crop_size[1] + dy, dx:crop_size[0] + dx].copy()

        if random.random() > 0.8:
            target = np.fliplr(target).copy()
            img = img.transpose(Image.FLIP_LEFT_RIGHT).copy()

    return img, target
class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, 
                 transform=None, train=False, seen=0, 
                 batch_size=1, num_workers=4,
                 suffix='', crop=False, train_size=(720, 720)):

        random.shuffle(root)
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.suffix = suffix
        self.crop = crop
        self.train_size = train_size

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.lines[index]

        img, target = load_data(img_path, self.train, self.suffix, self.crop, self.train_size)
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

