
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import skimage.external.tifffile as tif
import numpy as np

train_dataset_path = '../../HyRANK_satellite/TrainingSet'
val_dataset_path = '../../HyRANK_satellite/ValidationSet'

img = tif.imread(os.path.join(train_dataset_path, 'Loukia_GT.tif'))

unique, counts = np.unique(img, return_counts=True)

oc_dict = dict(zip(unique, counts))
print(oc_dict)