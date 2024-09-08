
import os
from easydict import EasyDict as edict
import time
import json
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
from csrnet import CSRNet
from mcnn import MCNN
from sanet import SANet
from vgg import VGG
from train_utils import train, validate, save_checkpoint
from utils import device
import sys

IS_DEBUG = False

def get_config(i, model_name, type):
    cfg = edict()
    cfg["train_json"] = f"./data_path/train_2.0_f{i}.json"  # path to train json
    cfg["test_json"] = f"./data_path/test_2.0_f{i}.json"  # path to test json
    cfg["suffix"] = type
    cfg["task"] =  f'{model_name}_2.0_f{i}_' + cfg["suffix"]  # task is to use
    cfg["pre"] = cfg["task"] + "checkpoint.pth.tar"  # path to the pretrained model
    cfg["start_epoch"] = 0  # Starting epoch (impact learning rate)
    cfg["epochs"] = 1 if IS_DEBUG else 500  # Epoch
    cfg["best_prect"] = 1e6  # Optimal accuracy
    cfg["original_lr"] = 1e-6  # Initial learning rate
    cfg["lr"] = 1e-6  # learning rate
    cfg["batch_size"] = 4  # batch_size
    cfg["decay"] = 1e-4  # Learning rate decay
    cfg["workers"] = 4  # Number of threads
    cfg["seed"] = time.time()  # Random seeds
    cfg["stand_by"] = 10
    cfg["print_freq"] = 10  # Print queue

    cfg.random_flip = True
    cfg.crop = False
    cfg.train_size = (720, 720)

    cfg.LR_TMAX = 10
    cfg.LR_COSMIN = 1e-6
    cfg.LR_DECAY_START = -1
    return cfg

def create_model(name):
    if name == 'csrnet':
        return CSRNet()
    elif name == 'mcnn':
        return MCNN()
    elif name == 'sanet':
        return SANet()
    else:
        return VGG()

def train_by_model_and_type(model_name, type):
    k = 2 if IS_DEBUG else 6
    for i in range(1, k, 1):
        cfg = get_config(i, model_name, type)
    
        torch.cuda.manual_seed(cfg["seed"])

        with open(cfg["train_json"], "r") as outfile:
            train_list = json.load(outfile)
            if IS_DEBUG:
                train_list = ['above0-00-40', 'above0-04-00']
        with open(cfg["test_json"], "r") as outfile:
            val_list = json.load(outfile)
            if IS_DEBUG:
                val_list = ['above0-07-30']

        model = create_model(model_name)
        # model = model.cuda()
        # criterion = nn.MSELoss().cuda()
        model = model.to(device)
        criterion = nn.MSELoss().to(device)

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.LR_TMAX, eta_min=cfg.LR_COSMIN)
        
        weight_save_dir = os.path.join('weights', cfg['task'])
        if not os.path.isdir(weight_save_dir):
            os.makedirs(weight_save_dir)


        prec1 = cfg['best_prec']

        history = []
        for epoch in range(cfg['start_epoch'], cfg['epochs']):
            train(train_list, model, criterion, optimizer, 
                  epoch, cfg['batch_size'], cfg['workers'],
                  cfg['suffix'], cfg.crop, cfg.train_size)
         
            prec1 = validate(val_list, model, criterion,
                             cfg['suffix'], cfg.crop, cfg.train_size)
            
            history.append(float(prec1))
            is_best = prec1 < cfg['best_prect']
            cfg['best_prect'] = min(prec1, cfg['best_prect'])
            print(' * best MAE {mae:.3f} '
                    .format(mae=cfg['best_prect']))
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': cfg['pre'],
                'state_dict': model.state_dict(),
                'best_prect': cfg['best_prect'],
                'optimizer': optimizer.state_dict(),
            }, is_best, cfg['task'], weight_save_dir, history=history)

            scheduler.step()


if __name__ == '__main__':
    types = ['ground_truth_dot', 'ground_truth_line', 'ground_truth_anisotropy_1_4']
    models = ['csrnet', 'mcnn', 'sanet', 'vgg']
    
    parameter = sys.argv[1:]
    if len(parameter) > 0 and (parameter[0] == 'debug' or parameter[0] == 'd'):
        IS_DEBUG = True
    
    for model_name in models:
        for type in types:
            train_by_model_and_type(model_name, type)