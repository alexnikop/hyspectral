'''
load train data, process them, index with in (imd_idx, x_coor, y_coor) 
tuples  and create batches when requested
'''

import os
import skimage.external.tifffile as tif
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import pickle
import random

class TrainDataLoader:
    def __init__(self,
                 pca_components=-1,
                 data_spatial_size=25,
                 dataset_folder = '../../HyRANK_satellite',
                 pca_load=False,
                 pca_save_folder='.',
                 no_classes=14,
                 train_mode='simple'):

        # when pca_load = true, pca algorithm is loaded from folder
        # and is not retrained
        self.pca_load = pca_load
        
        self.pca_components = pca_components
        self.pca_save_folder = pca_save_folder
        self.data_spatial_size = data_spatial_size
        self.margin = int((self.data_spatial_size - 1) / 2)
        self.no_classes = no_classes
        self.train_mode = train_mode

        # load data
        X, y = self.load_data(os.path.join(dataset_folder, 'TrainingSet'))

        self.train_fingerprints, self.val_fingerprints = self.generate_data_fingerprints(
            X, y)

        self.train_labels = y
        self.train_images = self.process_images(X)

    # shuffle fingerprint lists to diversify each epoch
    # train operation

    def shuffle_data(self):
        random.shuffle(self.train_fingerprints)
        random.shuffle(self.val_fingerprints)

    # load tif arrays from data folder

    def load_data(self, data_path):

        data_images = os.listdir(data_path)

        X = list()
        y = list()

        for img_name in data_images:
            img = tif.imread(os.path.join(data_path, img_name))
            if 'GT' in img_name:
                y.append(img)
            else:
                X.append(img)

        return X, y

    # preprocess train data by applying pca to reduce
    # spectral dmension and padding them

    def process_images(self, X):

        if self.pca_components > 0:
            X = self.pca(X)

        X_padded = self.pad_zeros(X, self.margin)

        return X_padded

    # reduces 176 spectral bands to
    # the number of self.pca_components

    def pca(self, X):

        # reshape and concat train images
        # along spectral dimension

        total_X = list()
        for xi in X:
            xi_reshaped = np.reshape(xi, (-1, xi.shape[2]))
            total_X.append(xi_reshaped)

        total_X = np.concatenate(total_X)

        # apply pca algorithm

        if not self.pca_load:
            pca = PCA(n_components=self.pca_components, whiten=True)
            pca_X = pca.fit_transform(total_X)

            with open('{}/pca.pkl'.format(self.pca_save_folder), 'wb') as f:
                pickle.dump(pca, f)
        else:
            with open('{}/pca.pkl'.format(self.pca_save_folder), 'rb') as f:
                pca = pickle.load(f)
            pca_X = pca.transform(total_X)

        # reconstruct original images to
        # preserve spatial dependancies

        offset = 0
        final_X = list()
        for xi in X:
            flat_dim = xi.shape[0] * xi.shape[1]
            xi_split = pca_X[offset:offset + flat_dim]
            xi_reduced = np.reshape(
                xi_split, (xi.shape[0], xi.shape[1], self.pca_components))
            final_X.append(xi_reduced)
            offset += flat_dim

        return final_X

    # pad zeros so that all the train data
    # will have the same dimensions

    def pad_zeros(self, X, margin=10):

        final_X = list()

        for xi in X:
            new_xi = np.zeros((xi.shape[0] + 2 * margin,
                               xi.shape[1] + 2 * margin, xi.shape[2]))
            x_offset = margin
            y_offset = margin
            new_xi[x_offset:xi.shape[0] + x_offset,
                   y_offset:xi.shape[1] + y_offset, :] = xi
            final_X.append(new_xi)

        return final_X

    # create data fingerprints (img_idx, coor_x, coor_y) so that
    # the algorithm can easily identify the requested data when 
    # creating a batch. This saves a lot of memory.

    def generate_data_fingerprints(self, X, y):

        fingerprints = list()

        # create fingerprints
        for idx, xi in enumerate(X):
            for c in range(xi.shape[0]):
                for r in range(xi.shape[1]):
                    fingerprint = (idx, c, r)
                    fingerprints.append(fingerprint)

        # train test split. use same random_state to make sure
        # that validation data aren't used in training when using 
        # multiple validators

        train_fingerprints, val_fingerprints = train_test_split(
            fingerprints, shuffle=True, test_size=0.17, random_state=7)

        return train_fingerprints, val_fingerprints

    # format batch_X, batch_y to be in a ready state for the 
    # cnn algorithm
    def final_batch_process(self, batch_X, batch_y):
        batch_X_arr = np.array(batch_X)
        batch_X_arr = np.expand_dims(batch_X_arr, axis=4)
        batch_y_arr = np.array(batch_y)
        batch_y_arr -= 1
        return batch_X_arr, batch_y_arr

    # augment data cube by applying a random transformation
    # and random gaussian noise
    def augment_cube(self, cube):

        randint = random.randint(0, 5)
        randoffset = random.uniform(0, 1)
        max_aug_noise_perc = 0.0025
        #randoffset = 0

        if randint == 1:
            cube = np.fliplr(cube)
        elif randint == 2:
            cube = np.flipud(cube)
        elif randint == 3:
            cube = np.rot90(cube, 1, (0, 1))
        elif randint == 4:
            cube = np.rot90(cube, 1, (1, 0))
        elif randint == 5:
            cube = np.rot90(cube, 2)

        cube += randoffset * max_aug_noise_perc * np.random.normal(size=(cube.shape))

        return cube

    # batch all data and return them for train or validation

    def batchify_data(self, batch_size, step='train'):

        if step == 'train':
            fingerprints = self.train_fingerprints
        else:
            fingerprints = self.val_fingerprints

        batch_X = list()
        batch_y = list()

        for data_idx, c, r in fingerprints:
            
            # if data point is unlabeled skip

            if self.train_labels[data_idx][c][r] == 0:
                continue
            
            # cut data cube
            data_cube = self.train_images[data_idx][c:c + 2 * self.margin + 1,
                                                    r:r + 2 * self.margin + 1]

            # apply data augmentation
            #if step == 'train':
            #    data_cube = self.augment_cube(data_cube)

            cube_label = self.train_labels[data_idx][c][r]

            batch_X.append(data_cube)
            batch_y.append(cube_label)

            # if batch is ready process it and return it

            if len(batch_X) == batch_size:
                batch_X_arr, batch_y_arr = self.final_batch_process(
                    batch_X, batch_y)
                batch_X = list()
                batch_y = list()
                yield batch_X_arr, batch_y_arr

        # return final batch if batch_length < batch_size
        batch_X_arr, batch_y_arr = self.final_batch_process(batch_X, batch_y)
        yield batch_X_arr, batch_y_arr


if __name__ == '__main__':

    TrainDataLoader(pca_components=30)
