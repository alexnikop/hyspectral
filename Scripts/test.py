from dataloader_test import TestDataLoader
from tensorflow.keras.models import load_model
import os
import numpy as np
import skimage.external.tifffile as tif
from PIL import Image
from utils import get_color_dict, create_folder


def test_model(test_imgs_path, model_folder, save_folder, batch_size=256):

    create_folder(save_folder)

    color_dict = get_color_dict()

    model_path = os.path.join(model_folder, 'best_model.h5')
    model = load_model(model_path)
    spatial_size = model.input.shape[1]

    loader = TestDataLoader(data_spatial_size=spatial_size,
                            pca_load_folder=model_folder)

    for img_name in os.listdir(test_imgs_path):

        img_path = os.path.join(test_imgs_path, img_name)
        test_img = tif.imread(img_path)
        img_preds = list()

        for batch in loader.batchify_image(test_img, batch_size):
            pred = model(batch, training=False)
            pred = np.argmax(pred, axis=1)
            img_preds += pred.tolist()

        img_pred = np.zeros((test_img.shape[0], test_img.shape[1], 3))
        img_preds = np.reshape(np.array(img_preds),
                               [test_img.shape[0], test_img.shape[1]])

        for cls_idx in range(14):
            img_pred[img_preds == cls_idx] = color_dict[cls_idx]

        img_pred = img_pred.astype(np.uint8)

        pil_image = Image.fromarray(img_pred, 'RGB')
        pil_image.save('{}/{}.png'.format(save_folder, img_name.split('.')[0]))


if __name__ == '__main__':

    test_imgs_path = '../../HyRANK_satellite/ValidationSet'
    model_folder = '/home/alexnikop/up2metric/hyspectral/Scripts/model_size_9_spec_14'
    save_folder = 'results'

    test_model(test_imgs_path, model_folder, save_folder)