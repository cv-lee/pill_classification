import os
import sys

import torch
from torch.optim import Adam

from .resnet import *
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils import *

def load_model(args, mode):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.data == 'shape':
        class_num = 11
    elif args.data == 'color1' or args.data == 'color2':
        class_num = 16
    elif args.data == 'all':
        class_num = (11,16,16)
    else:
        raise ValueError('args.data ERROR')
    model = resnet18(num_classes=class_num, drop_rate=args.drop_rate)
    if mode == 'TRAIN':
        optimizer = Adam(model.parameters(), lr=args.lr)
        resume = args.resume
    elif mode == 'TEST' or mode=='VALID':
        optimizer = None
        resume = True
    else:
        raise ValueError('InValid Flag in load_model')

    if resume:
        checkpoint = Checkpoint(model, optimizer)
        checkpoint.load(args.ckpt_path)
        best_loss = checkpoint.best_loss
        start_epoch = checkpoint.epoch+1
    else:
        best_loss = 9999
        start_epoch = 1

    if device == 'cuda':
        model.cuda()
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark=True
    return model, optimizer, best_loss, start_epoch

