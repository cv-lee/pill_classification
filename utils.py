import torch
import cv2
import sys
import time
import pdb
import numpy as np
import os


from skimage import color, segmentation, color
from skimage.segmentation import slic
from skimage.future import graph
from skimage.measure import regionprops
from skimage.filters import gaussian

# Paths Init
IMG_ROOT = './image'
MASK_ROOT = './mask'

# Parameters Init
_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time



def get_loss(pred, target):
    if type(label) == type(tuple):
        shape_loss = F.cross_entropy(pred[0], target)
        color1_loss = F.cross_entropy(pred[1], target)
        color2_loss = F.cross_entropy(pred[2], target)
        return shape_loss, color1_loss, color2_loss
    else:
        return F.cross_entropy(pred, target)

def pill_mask(img_root, mask_root):
    ''' Mask Pill Mask

        -Reference Paper-
        http://www.scitepress.org/Papers/2017/61358/61358.pdf
    '''

    img_list = os.listdir(img_root)
    img_list = sorted(img_list, key=lambda x: int(os.path.splitext(x)[0]))
    for i, img_name in enumerate(img_list):
        print(i)
        img = cv2.imread(os.path.join(img_root, img_name))
        shear = int(img.shape[0] * 0.12)
        pad_img = np.zeros((shear, img.shape[1]))
        img = img[:(img.shape[0]-shear), :, :]

        # Gaussian Smoothing Filter
        img = (gaussian(img, sigma=2, multichannel=True)*255).astype(np.uint8)

        # SLIC (Simple Linear Iterative Clustering)
        labels = slic(img, n_segments=150, compactness=12, max_iter=10)
        labels = labels + 1
        regions = regionprops(labels)

        # Create RAG(Region Adjacency Graph)
        rag = graph.rag_mean_color(img, labels)
        for region in regions:
            rag.node[region['label']]['centroid'] = region['centroid']

        # Post-processing
        labels = graph.cut_threshold(labels, rag, 29)
        background = np.argmax(np.bincount(labels.flatten()))
        labels[labels!=background] = 255
        labels[labels==background] = 0
        labels = np.concatenate((labels, pad_img),axis=0)

        cv2.imwrite(os.path.join(mask_root, img_name), labels)

def progress_bar(current, total, msg=None):
    ''' Source Code from 'kuangliu/pytorch-cifar'
        (https://github.com/kuangliu/pytorch-cifar/blob/master/utils.py)
    '''
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    ''' Source Code from 'kuangliu/pytorch-cifar'
        (https://github.com/kuangliu/pytorch-cifar/blob/master/utils.py)
    '''
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


class Checkpoint:
    def __init__(self, model, optimizer=None, epoch=0, best_loss=9999):
        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch
        self.best_loss = best_loss

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state"])
        self.epoch = checkpoint["epoch"]
        self.best_loss = checkpoint["best_loss"]
        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])

    def save(self, path):
        state_dict = self.model.module.state_dict()
        torch.save({"model_state": state_dict,
                    "optimizer_state": self.optimizer.state_dict(),
                    "epoch": self.epoch,
                    "best_loss": self.best_loss}, path)

