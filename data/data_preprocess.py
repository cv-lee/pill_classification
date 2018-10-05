import pandas as pd
import os
import urllib
import pdb
import cv2

LABEL_PATH = './label/label.xls'
IMAGE_ROOT = './image'

def crawling(label_path, output_root):
    ''' Download images based on image link
    '''
    xls = pd.read_excel(label_path)
    for i in range(len(xls['link'])):
        link = xls['link'][i]
        name = xls['No'][i]
        img_name = os.path.join(output_root, str(name)) + '.jpg'
        if not(isinstance(link, str)):
            continue
        if link.split(':')[0] == 'http':
            urllib.request.urlretrieve(link, img_name)


def imgCrop(img_root):
    ''' Croping and Resize images to 1024*512*3
    '''
    for idx, img_name in enumerate(os.listdir(img_root)):
        path = os.path.join(img_root, img_name)
        img = cv2.imread(path)
        img = cv2.resize(img, (1300,710))
        img = img[10:10+650, :, :]
        img = cv2.resize(img, (1024,512))
        cv2.imwrite(path, img)
        print(idx)

def sizeCheck(img_root):
    size_list = []
    for img_name in os.listdir(img_root):
        path = os.path.join(img_root, img_name)
        img = cv2.imread(path)
        if isinstance(img, type(None)):
            print(path)
        else:
            size_list.append(img.shape)
    print(set(size_list))

#crawling(LABEL_PATH, IMAGE_ROOT)
#imgCrop(IMAGE_ROOT)
#sizeCheck(IMAGE_ROOT)
