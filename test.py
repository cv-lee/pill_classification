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

def test(args):

    # Device Init
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cudnn.benchmark = True

    # Data Load
    testloader = data_loader(args, mode='TEST')

    # Model Load
    resnet, _, _, _ = load_model(args, mode='TEST')

    resnet.eval()
    torch.set_grad_enabled(False)
    loss = [0, 0, 0, 0] # shape,color1,color2,total
    acc = [0, 0, 0, 0] # shape,color1,color2,total
    aleatoric_val = 0
    epistemic_val = 0
    for idx, (imgs, labels, paths) in enumerate(testloader):
        batch_size = imgs.shape[0]
        if type(labels) == list:
            imgs, shape, color1, color2 = imgs.to(device), labels[0].to(device),\
                                         labels[1].to(device), labels[2].to(device)
            outputs = []
            for i in range(args.iter_num):
                output = resnet(imgs)
                outputs.append(output)

                shape_loss = F.cross_entropy(output[0], shape)
                color1_loss = F.cross_entropy(output[1], color1)
                color2_loss = F.cross_entropy(output[2], color2)
                batch_loss = config.shape_loss_weight*shape_loss +\
                             config.color1_loss_weight*color1_loss +\
                             config.color2_loss_weight*color2_loss
                loss[0] += float(shape_loss)/args.iter_num
                loss[1] += float(color1_loss)/args.iter_num
                loss[2] += float(color2_loss)/args.iter_num
                loss[3] += float(batch_loss)/args.iter_num
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
            imgs, labels = imgs.to(device), labels.to(device)
            batch_loss = 0
            for i in range(args.iter_num):
                pred = resnet(imgs)
                batch_loss += float(F.cross_entropy(pred, labels))/args.iter_num
                pred = pred.cpu().detach().numpy()
                pred = np.expand_dims(pred, axis=0)
                if i == 0:
                    preds = pred
                else:
                    preds = np.concatenate((preds, pred), axis=0)
            prediction = np.mean(preds, axis=0)
            aleatoric = np.mean(preds*(1-preds), axis=0)
            epistemic = np.mean(preds**2, axis=0) - np.mean(preds, axis=0)**2
            prediction = prediction.argmax(axis=1)
            labels = labels.cpu().detach().numpy()

            pred_aleatoric = []
            pred_epistemic = []
            for i in range(batch_size):
                pred_aleatoric.append(aleatoric[i, prediction[i]])
                pred_epistemic.append(epistemic[i, prediction[i]])
            aleatoric_val += np.array(pred_aleatoric).sum() / len(pred_aleatoric)
            epistemic_val += np.array(pred_epistemic).sum() / len(pred_epistemic)

            acc[3] += float(((prediction==labels).astype(np.int).sum())/batch_size)
            loss[3] += float(batch_loss)
            progress_bar(idx, len(testloader), 'loss: %.4f, acc: %.4f. alea: %.4f, epis: %.4f'
                %((loss[3]/(idx+1)), acc[3]/(idx+1), aleatoric_val/(idx+1), epistemic_val/(idx+1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default="shape",
                        help="data to train(shape, color1, color2, all)")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="The batch size to load the data")
    parser.add_argument("--iter_num", type=int, default=20,
                        help="The training epochs to run.")
    parser.add_argument("--drop_rate", type=float, default=0.25,
                        help="Drop-out rate for uncertainty model")
    parser.add_argument("--img_root", type=str, default="./data/image",
                        help="The directory containing the training image dataset.")
    parser.add_argument("--label_path", type=str, default="./data/label/label.xls",
                        help="The directory containing the training label datgaset")
    parser.add_argument("--ckpt_path", type=str, default="./checkpoint/resnet.tar",
                        help="The directory containing the training label datgaset")
    args = parser.parse_args()

    test(args)
