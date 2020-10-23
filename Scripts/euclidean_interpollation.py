'''
Generates labels for unlabeled data, by using the average band vector of each class, 
which is calculated using the labeled data. The script calculates the euclidean 
distance of each unlabeled band vector with each class's average vector and sets the 
label that corresponds to the minimum distance.
'''

import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import skimage.external.tifffile as tif
import numpy as np
from utils import get_color_dict

train_dataset_path = '../../HyRANK_satellite/TrainingSet'
save_dataset_path = '../../HyRANK_satellite/customTrainingSet'
#val_dataset_path = '../../HyRANK_satellite/customValidationSet'

train_data = os.listdir(train_dataset_path)
train_images = [x for x in train_data if 'GT' not in x]
trian_labels = [x for x in train_data if "GT" in x]

total_dict = dict()

for img_path, lab_path in zip(train_images, trian_labels):

    img = tif.imread(os.path.join(train_dataset_path, img_path))
    lab = tif.imread(os.path.join(train_dataset_path, lab_path))

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if (lab[x][y] > 0 and lab[x][y] not in total_dict.keys()):
                total_dict[lab[x][y]] = list()
            
            if lab[x][y] > 0:
                total_dict[lab[x][y]].append(img[x][y])


avg_dict = dict()

for k,v in total_dict.items():
    avg_dict[k] = np.mean(v, axis=0)


def calc_min_distance(vector):
    minkey = 0
    mindist = None

    for k,v in avg_dict.items():
        dist = np.linalg.norm(vector - v)
        if (minkey == 0 or dist < mindist):
            minkey = k
            mindist = dist
    
    return minkey, mindist


color_dict = get_color_dict()

for img_path in train_images:

    img = tif.imread(os.path.join(train_dataset_path, img_path))

    label_img = np.zeros((img.shape[0], img.shape[1]))
    class_img = np.zeros((img.shape[0], img.shape[1], 3))

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            minkey, mindist = calc_min_distance(img[x][y])
            label_img[x][y] = minkey
            chosen_color = color_dict[minkey - 1]
            class_img[x][y][0] = chosen_color[0]
            class_img[x][y][1] = chosen_color[1]
            class_img[x][y][2] = chosen_color[2]

    tif.imsave('{}/{}_GT.tif'.format(save_dataset_path, img_path.split('.')[0]), label_img)
    class_img = class_img.astype(np.uint8)

    save_img = Image.fromarray(class_img, 'RGB')
    save_img.save('{}.png'.format(img_path.split('.')[0]))