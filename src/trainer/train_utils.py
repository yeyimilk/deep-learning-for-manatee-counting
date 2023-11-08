
from utils import AverageMeter, listDataset
import torch
from torchvision import transforms
from torch.autograd import Variable
import time
from utils import device

def validate(val_list, model, criterion, suffix, crop, train_size):
    print('begin test')
    test_loader = torch.utils.data.DataLoader(
        listDataset(val_list,
                    shuffle=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225]),
                    ]), 
                    train=False,
                    suffix=suffix, crop=crop, train_size=train_size),
        batch_size=1)

    model.eval()

    mae = 0

    for _, (img, target) in enumerate(test_loader):
        # img = img.cuda()
        img = img.to(device)
        img = Variable(img)
        output = model(img)
        target_sum = target.sum().type(torch.FloatTensor).to(device)
        # mae += abs(output.data.sum() - target.sum().type(torch.FloatTensor).cuda())
        mae += abs(output.data.sum() - target_sum)

    mae = mae / len(test_loader)
    print(' * MAE {mae:.3f} '.format(mae=mae))

    return mae

def train(train_list, model, criterion, optimizer, epoch, batch_size, num_workers, suffix, crop, train_size):
    # loss, training time, data loading time
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # read training data (including data enhancement)
    train_loader = torch.utils.data.DataLoader(
        listDataset(train_list,
                    shuffle=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225]),
                    ]),
                    train=True,
                    seen=model.seen,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    suffix=suffix, crop=crop, train_size=train_size))
    print('epoch %d, processed %d samples, lr %.4f' %
          (epoch, epoch * len(train_loader.dataset), optimizer.param_groups[0]['lr'] * 10000))

    model.train()
    end = time.time()

    for i, (img, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # img = img.cuda()
        img = img.to(device)
        img = Variable(img)
        output = model(img)

        # target = target.type(torch.FloatTensor).unsqueeze(1).cuda()
        target = target.type(torch.FloatTensor).unsqueeze(1).to(device)
        target = Variable(target)

        loss = criterion(output, target)

        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # if i % cfg['print_freq'] == 0 and 0:
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #     .format(
        #         epoch, i, len(train_loader), batch_time=batch_time,
        #         data_time=data_time, loss=losses))
