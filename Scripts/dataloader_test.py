import os
import skimage.external.tifffile as tif
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import pickle


class TestDataLoader:
    def __init__(self,
                 data_spatial_size=25,
                 pca_load_folder='.',
                 base_path='../../HyRANK_satellite'):

        self.data_spatial_size = data_spatial_size

        with open('{}/pca.pkl'.format(pca_load_folder), 'rb') as f:
            self.pca = pickle.load(f)

    def convert_pca(self, X):

        # reshape along spectral dimension

        X_flat = np.reshape(X, (-1, X.shape[2]))
        X_flat_pca = self.pca.transform(X_flat)
        X_pca = np.reshape(X_flat_pca,
                           (X.shape[0], X.shape[1], X_flat_pca.shape[1]))

        return X_pca

    # pad images with zeros so that we can
    # cut same dimension cubes for edge cases

    def pad_zeros(self, X, margin=10):
        newX = np.zeros(
            (X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
        x_offset = margin
        y_offset = margin
        newX[x_offset:X.shape[0] + x_offset,
             y_offset:X.shape[1] + y_offset, :] = X
        return newX

    def batchify_image(self, test_img, batch_size):

        margin = int((self.data_spatial_size - 1) / 2)

        test_img = self.convert_pca(test_img)
        test_img_padded = self.pad_zeros(test_img, margin)

        batch_lst = list()

        for c in range(margin, test_img_padded.shape[0] - margin):
            for r in range(margin, test_img_padded.shape[1] - margin):

                cube = test_img_padded[c - margin:c + margin + 1,
                                       r - margin:r + margin + 1]

                batch_lst.append(cube)

                if len(batch_lst) == batch_size:
                    batch_arr = np.array(batch_lst)
                    batch_arr = np.expand_dims(batch_arr, axis=4)
                    batch_lst = list()
                    yield batch_arr

        batch_arr = np.array(batch_lst)
        batch_arr = np.expand_dims(batch_arr, axis=4)

        yield batch_arr


if __name__ == '__main__':

    loader = TestDataLoader(data_spatial_size=15)
    img_path = '/home/alexnikop/up2metric/HyRANK_satellite/ValidationSet/Erato.tif'
    batch_size = 256

    counter = 0
    for batch in loader.batchify_image(img_path, batch_size):
        print(batch.shape)
        counter += 1
