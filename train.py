import argparse
import pdb

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from dataset import *
from models import *
from utils import *


def train(args):

    # Device Init
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cudnn.benchmark = True

    # Data Load
    trainloader = data_loader(args, mode='TRAIN')
    validloader = data_loader(args, mode='VALID')

    # Model Load
    resnet, optimizer, best_loss, start_epoch =\
        load_model(args, mode='TRAIN')

    # Loss Init
    for epoch in range(start_epoch, args.epochs + 1):
        # Train Model
        print('Epoch: {}\n  >> Train'.format(epoch))
        resnet.train(True)
        loss = 0
        acc = 0
        lr = args.lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        for idx, (img, label, img_path) in enumerate(trainloader):
            img, label = img.to(device), label.to(device)
            output = resnet(img)
            batch_loss = F.cross_entropy(output, label)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            output = (output.cpu().detach().numpy()).argmax(axis=1)
            label = label.cpu().detach().numpy()
            total = output.shape[0]
            correct = (output == label).astype(np.int).sum()
            acc += float(correct/total)
            loss += float(batch_loss)
            progress_bar(idx, len(trainloader), 'Loss: %.5f, Acc: %.5f'
                    %((loss/(idx+1)), acc/(idx+1)))

        print('\n\n  >>Validation')
        resnet.eval()
        loss = 0
        acc = 0
        for idx, (img, label, img_path) in enumerate(trainloader):
            img, label = img.to(device), label.to(device)
            output = resnet(img)
            output = (output.cpu().detach().numpy()).argmax(axis=1)
            label = label.cpu().detach().numpy()
            total = output.shape[0]
            correct = (output == label).astype(np.int).sum()
            acc += float(correct/total)
            loss += float(batch_loss)
            progress_bar(idx, len(trainloader), 'Loss: %.5f, Acc: %.5f'
                    %((loss/(idx+1)), acc/(idx+1)))

        loss /= (idx+1)
        if loss < best_loss:
            checkpoint = Checkpoint(resnet, optimizer, epoch, loss)
            checkpoint.save(args.ckpt_path)
            best_loss = loss
            print("Saving...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=bool, default=False,
                        help="Model Trianing resume.")
    parser.add_argument("--data", type=str, default="shape",
                        help="data to train(shape, color1, color2)")
    parser.add_argument("--batch_size", type=int, default=12,
                        help="The batch size to load the data")
    parser.add_argument("--epochs", type=int, default=150,
                        help="The training epochs to run.")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate to use in training")
    parser.add_argument("--img_root", type=str, default="./data/image",
                        help="The directory containing the training image dataset.")
    parser.add_argument("--label_path", type=str, default="./data/label/label.xls",
                        help="The directory containing the training label datgaset")
    parser.add_argument("--ckpt_path", type=str, default="./checkpoint/unet.tar",
                        help="The directory containing the training label datgaset")
    args = parser.parse_args()

    train(args)
