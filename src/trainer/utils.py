import random
from torch.utils.data import Dataset

class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, 
                 transform=None, train=False, 
                 seen=0, batch_size=1,
                 num_workers=4):
        
        random.shuffle(root)

        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
       
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.load_data = lambda x: (x, self.train) # TODO

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.lines[index]

        img, target = self.load_data(img_path, self.train)

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
        

