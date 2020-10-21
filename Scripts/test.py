
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import skimage.external.tifffile as tif
import numpy as np

train_dataset_path = '../../HyRANK_satellite/TrainingSet'
val_dataset_path = '../../HyRANK_satellite/ValidationSet'

img = tif.imread(os.path.join(train_dataset_path, 'Loukia.tif'))
img_GT = tif.imread(os.path.join(train_dataset_path, 'Loukia_GT.tif'))


print(img.shape)
print(img_GT.shape)

total_dict = dict()

for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        if (img_GT[x][y] > 0 and img_GT[x][y] not in total_dict.keys()):
            total_dict[img_GT[x][y]] = list()
        
        if img_GT[x][y] > 0:
            total_dict[img_GT[x][y]].append(img[x][y])


avg_dict = dict()

for k,v in total_dict.items():
    avg_dict[k] = np.mean(v, axis=0)



for k, v in avg_dict.items():
    print(k, v.shape)


color_dict = dict()
color_dict[1] = (255,2,4)
color_dict[2] = (192,52,111)
color_dict[3] = (235, 233, 36)
color_dict[4] = (244, 126, 39)
color_dict[5] = (235, 181, 71)
color_dict[6] = (0, 173, 48)
color_dict[7] = (0, 72, 24)
color_dict[8] = (90, 30, 106)
color_dict[9] = (113, 122, 50)
color_dict[10] = (192, 173, 69)
color_dict[11] = (214, 236, 101)
color_dict[12] = (172, 183, 187)
color_dict[13] = (39, 76, 212)
color_dict[14] = (96, 229, 254)



def calc_min_distance(vector):
    minkey = 0
    mindist = None

    for k,v in avg_dict.items():
        dist = np.linalg.norm(vector - v)
        if (minkey == 0 or dist < mindist):
            minkey = k
            mindist = dist
    
    return minkey, mindist

class_img = np.zeros((img.shape[0], img.shape[1], 3))

for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        minkey, mindist = calc_min_distance(img[x][y])
        chosen_color = color_dict[minkey]
        class_img[x][y][0] = chosen_color[0]
        class_img[x][y][1] = chosen_color[1]
        class_img[x][y][2] = chosen_color[2]


print(np.unique(class_img))

save_img = Image.fromarray(class_img, 'RGB')
save_img.save('test.png')
'''
unique, counts = np.unique(img, return_counts=True)
oc_dict = dict(zip(unique, counts))
print(oc_dict)
'''