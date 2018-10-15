import argparse
import pdb

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from dataset import *
from models import *
from utils import *
import config

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
        torch.set_grad_enabled(True)
        loss = [0, 0, 0, 0] # shape,color1,color2,total
        acc = [0, 0, 0, 0] # shape,color1,color2,total
        lr = args.lr * (0.5 ** (epoch // 5))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        for idx, (img, label, img_path) in enumerate(trainloader):
            batch_size = img.shape[0]
            if type(label) == list:
                img, shape, color1, color2 = img.to(device), label[0].to(device),\
                                             label[1].to(device), label[2].to(device)
                output = resnet(img)
                shape_loss = F.cross_entropy(output[0], shape)
                color1_loss = F.cross_entropy(output[1], color1)
                color2_loss = F.cross_entropy(output[2], color2)
                batch_loss = config.shape_loss_weight*shape_loss +\
                             config.color1_loss_weight*color1_loss +\
                             config.color2_loss_weight*color2_loss
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                loss[0] += float(shape_loss)
                loss[1] += float(color1_loss)
                loss[2] += float(color2_loss)
                loss[3] += float(batch_loss)

                pred_shape = (output[0].cpu().detach().numpy()).argmax(axis=1)
                pred_color1 = (output[1].cpu().detach().numpy()).argmax(axis=1)
                pred_color2 = (output[2].cpu().detach().numpy()).argmax(axis=1)
                shape = shape.cpu().detach().numpy()
                color1 = color1.cpu().detach().numpy()
                color2 = color2.cpu().detach().numpy()
                shape_acc = float(((shape==pred_shape).astype(np.int).sum())/batch_size)
                acc[0] += shape_acc
                color1_acc = float(((color1==pred_color1).astype(np.int).sum())/batch_size)
                acc[1] += color1_acc
                color2_acc = float(((color2==pred_color2).astype(np.int).sum())/batch_size)
                acc[2] += color2_acc
                acc[3] += shape_acc * color1_acc * color2_acc
                progress_bar(idx, len(trainloader), config.multi_print_format
                            %(loss[0]/(idx+1), loss[1]/(idx+1), loss[2]/(idx+1), loss[3]/(idx+1),
                            acc[0]/(idx+1), acc[1]/(idx+1), acc[2]/(idx+1), acc[3]/(idx+1)))
            else:
                img, label = img.to(device), label.to(device)
                output = resnet(img)
                batch_loss = F.cross_entropy(output, label)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                output = (output.cpu().detach().numpy()).argmax(axis=1)
                label = label.cpu().detach().numpy()
                acc[3] += float(((output==label).astype(np.int).sum())/batch_size)
                loss[3] += float(batch_loss)
                progress_bar(idx, len(trainloader), 'loss: %.5f, acc: %.5f'
                             %((loss[3]/(idx+1)), acc[3]/(idx+1)))

        print('\n\n  >>Validation')
        resnet.eval()
        torch.set_grad_enabled(False)
        loss = [0, 0, 0, 0] # shape,color1,color2,total
        acc = [0, 0, 0, 0] # shape,color1,color2,total
        for idx, (img, label, img_path) in enumerate(trainloader):
            batch_size = img.shape[0]
            if type(label) == list:
                img, shape, color1, color2 = img.to(device), label[0].to(device),\
                                             label[1].to(device), label[2].to(device)
                output = resnet(img)
                shape_loss = F.cross_entropy(output[0], shape)
                color1_loss = F.cross_entropy(output[1], color1)
                color2_loss = F.cross_entropy(output[2], color2)
                batch_loss = config.shape_loss_weight*shape_loss +\
                             config.color1_loss_weight*color1_loss +\
                             config.color2_loss_weight*color2_loss

                loss[0] += float(shape_loss)
                loss[1] += float(color1_loss)
                loss[2] += float(color2_loss)
                loss[3] += float(batch_loss)

                pred_shape = (output[0].cpu().detach().numpy()).argmax(axis=1)
                pred_color1 = (output[1].cpu().detach().numpy()).argmax(axis=1)
                pred_color2 = (output[2].cpu().detach().numpy()).argmax(axis=1)
                shape = shape.cpu().detach().numpy()
                color1 = color1.cpu().detach().numpy()
                color2 = color2.cpu().detach().numpy()
                shape_acc = float(((shape==pred_shape).astype(np.int).sum())/batch_size)
                acc[0] += shape_acc
                color1_acc = float(((color1==pred_color1).astype(np.int).sum())/batch_size)
                acc[1] += color1_acc
                color2_acc = float(((color2==pred_color2).astype(np.int).sum())/batch_size)
                acc[2] += color2_acc
                acc[3] += shape_acc * color1_acc * color2_acc
                progress_bar(idx, len(trainloader), config.multi_print_format
                            %(loss[0]/(idx+1), loss[1]/(idx+1), loss[2]/(idx+1), loss[3]/(idx+1),
                            acc[0]/(idx+1), acc[1]/(idx+1), acc[2]/(idx+1), acc[3]/(idx+1)))
            else:
                img, label = img.to(device), label.to(device)
                output = resnet(img)
                output = (output.cpu().detach().numpy()).argmax(axis=1)
                label = label.cpu().detach().numpy()
                correct = (output==label).astype(np.int).sum()
                acc[3] += float(correct/batch_size)
                loss[3] += float(batch_loss)
                progress_bar(idx, len(trainloader), 'Loss: %.5f, Acc: %.5f'
                            %((loss[3]/(idx+1)), acc[3]/(idx+1)))

        total_loss = loss[3]/(idx+1)
        if total_loss < best_loss:
            checkpoint = Checkpoint(resnet, optimizer, epoch, total_loss)
            checkpoint.save(args.ckpt_path)
            best_loss = total_loss
            print("Saving...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=bool, default=False,
                        help="Model Trianing resume.")
    parser.add_argument("--data", type=str, default="shape",
                        help="data to train(shape, color1, color2, all)")
    parser.add_argument("--batch_size", type=int, default=20,
                        help="The batch size to load the data")
    parser.add_argument("--epochs", type=int, default=150,
                        help="The training epochs to run.")
    parser.add_argument("--drop_rate", type=float, default=0.25,
                        help="Drop-out rate for uncertainty model")
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
