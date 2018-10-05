import pandas as pd
import os
import urllib
import pdb


LABEL_PATH = './label/label.xls'
IMAGE_ROOT = './image'

def download(data_path, output_root):
    xls = pd.read_excel(data_path)
    for i in range(len(xls['link'])):
        print(i)
        link = xls['link'][i]
        name = xls['No'][i]
        img_name = os.path.join(output_root, str(name)) + '.jpg'
        if not(isinstance(link, str)):
            continue
        if link.split(':')[0] == 'http':
            urllib.request.urlretrieve(link, img_name)


download(LABEL_PATH, IMAGE_ROOT)
