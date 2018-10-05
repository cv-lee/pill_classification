''' < Reference Paper >
    http://www.scitepress.org/Papers/2017/61358/61358.pdf
'''

import cv2
import pdb
import numpy as np
import os

from skimage import color, segmentation, color
from skimage.segmentation import slic
from skimage.future import graph
from skimage.measure import regionprops
from skimage.filters import gaussian

IMG_ROOT = './image'
MASK_ROOT = './mask'


def pill_mask(img_root, mask_root):
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



pill_mask(IMG_ROOT, MASK_ROOT)
