from dataloader_test import TestDataLoader
from tensorflow.keras.models import load_model
import os
import numpy as np
import skimage.external.tifffile as tif
from PIL import Image
from utils import get_color_dict, create_folder
import argparse

# test new images and predict their classification maps

def test_model(test_imgs_path, model_folder, save_folder, batch_size=256):

    create_folder(save_folder)

    # get color dict for visualization
    color_dict = get_color_dict()

    # load model
    model_path = os.path.join(model_folder, 'best_model.h5')
    model = load_model(model_path)
    spatial_size = model.input.shape[1]

    # create data loader for test image
    loader = TestDataLoader(data_spatial_size=spatial_size,
                            pca_load_folder=model_folder)

    # for each image predict
    for img_name in os.listdir(test_imgs_path):
        
        img_path = os.path.join(test_imgs_path, img_name)
        test_img = tif.imread(img_path)
        img_preds = list()

        # predict image in batches
        for batch in loader.batchify_image(test_img, batch_size):
            pred = model(batch, training=False)
            pred = np.argmax(pred, axis=1) + 1
            img_preds += pred.tolist()

        # create label map and visualization image 
        vis_img_pred = np.zeros((test_img.shape[0], test_img.shape[1], 3))
        img_preds = np.reshape(np.array(img_preds),
                               [test_img.shape[0], test_img.shape[1]])

        # paint visualization image
        for cls_idx in range(1, 15):
            vis_img_pred[img_preds == cls_idx] = color_dict[cls_idx-1]

        vis_img_pred = vis_img_pred.astype(np.uint8)
        img_preds = img_preds.astype(np.uint8)

        pil_image = Image.fromarray(vis_img_pred, 'RGB')
        pil_image.save('{}/{}.png'.format(save_folder, img_name.split('.')[0]))
        tif.imsave('{}/{}.tif'.format(save_folder, img_name.split('.')[0]), img_preds)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', help='specify images_dir')
    parser.add_argument('--model_dir', help='specify model_dir')
    parser.add_argument('--batch_size', type=int, default=256, help='specify batch size')

    args = parser.parse_args()

    if args.model_dir is None:
        print('You need to specify model_dir')
        exit(1)

    if args.images_dir is None:
        print('You need to specify images_dir')
        exit(1)

    test_imgs_path = args.images_dir
    model_path = args.model_dir
    batch_size = args.batch_size

    save_folder = 'results'
    create_folder(save_folder) 


    test_model(test_imgs_path, model_path, save_folder, batch_size=batch_size)