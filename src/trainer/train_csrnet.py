
import os
from easydict import EasyDict as edict
import time
import json
import shutil
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
from csrnet import CSRNet
from train_utils import train, validate

def main():
    for f in range(1, 6, 1):
        cfg = edict()
        cfg["train_json"] = f"./data_path/train_2.0_f{str(f)}.json"  # path to train json
        cfg["test_json"] = f"./data_path/test_2.0_f{str(f)}.json"  # path to test json
        cfg["suffix"] = 'ground_truth_dot'
        cfg["task"] = "CSRNet" + f'_2.0_f{str(f)}_' + cfg["suffix"]  # task is to use
        cfg["pre"] = cfg["task"] + "checkpoint.pth.tar"  # path to the pretrained model
        cfg["start_epoch"] = 0  # Starting epoch (impact learning rate)
        cfg["epochs"] = 500  # Epoch
        cfg["best_prec1"] = 1e6  # Optimal accuracy
        cfg["original_lr"] = 1e-4  # Initial learning rate
        cfg["lr"] = 1e-4  # learning rate
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

        torch.cuda.manual_seed(cfg["seed"])

        with open(cfg["train_json"], "r") as outfile:
            train_list = json.load(outfile)
        with open(cfg["test_json"], "r") as outfile:
            val_list = json.load(outfile)

        model = CSRNet()
        model = model.cuda()
        criterion = nn.MSELoss().cuda()

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.LR_TMAX, eta_min=cfg.LR_COSMIN)
        weight_save_dir = os.path.join('weights', cfg['task'])

        if not os.path.isdir(weight_save_dir):
            os.makedirs(weight_save_dir)

        def save_checkpoint(state, is_best, task_id, filename='checkpoint.pth.tar', history=None):
            ckpt_path = os.path.join(weight_save_dir, task_id + filename)
            torch.save(state, ckpt_path)
            
            with open(os.path.join(weight_save_dir, "history.json"), "w") as f:
                json.dump(history, f)
                
            if is_best:
                if float(state['best_prec1']) < 4.0:
                    mae = str('{mae:.3f}'.format(mae=state['best_prec1']))
                    best_path = os.path.join(weight_save_dir, task_id + '_epoch_' + str(state['epoch']) + '_mae_' + mae + '.pth.tar')
                    shutil.copyfile(ckpt_path, best_path)


        prec1 = cfg['best_prec1']

        history = []
        for epoch in range(cfg['start_epoch'], cfg['epochs']):
            train(train_list, model, criterion, optimizer, epoch, cfg['batch_size'], cfg['workers'])
         
            prec1 = validate(val_list, model, criterion)
            history.append(float(prec1))
            is_best = prec1 < cfg['best_prec1']
            cfg['best_prec1'] = min(prec1, cfg['best_prec1'])
            print(' * best MAE {mae:.3f} '
                    .format(mae=cfg['best_prec1']))
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': cfg['pre'],
                'state_dict': model.state_dict(),
                'best_prec1': cfg['best_prec1'],
                'optimizer': optimizer.state_dict(),
            }, is_best, cfg['task'], history=history)

            scheduler.step()

if __name__ == '__main__':
    main()