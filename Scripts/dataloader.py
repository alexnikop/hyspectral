import os
import skimage.external.tifffile as tif
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

class TrainDataLoader:
    def __init__(self, pca_components = -1, data_spatial_size = 25, base_path = '../../HyRANK_satellite'):
        
        self.train_offset = 0
        self.val_offset = 0

        self.pca_components = pca_components
        self.data_spatial_size = data_spatial_size

        self.train_data, self.val_data = self.process_data(base_path)
        self.shuffle_data()


    # shuffle train, val data
    def shuffle_data(self):
        
        X_train, y_train = self.train_data
        X_val, y_val = self.val_data

        # shuffle train data
        train_idxes = np.random.permutation(X_train.shape[0])

        X_train = X_train[train_idxes]
        y_train = y_train[train_idxes]


        # shuffle val data
        val_idxes = np.random.permutation(X_val.shape[0])

        X_val = X_val[val_idxes]
        y_val = y_val[val_idxes]
        
        self.train_data = X_train, y_train
        self.val_data = X_val, y_val

    # preprocess train data and extract train / test splits

    def process_data(self, base_path):

        X, y = self.load_data(os.path.join(base_path, 'TrainingSet'))

        if self.pca_components > 0:
            X = self.pca(X)

        X, y = self.create_train_cube_data(X, y)
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train = np.expand_dims(X_train, axis=4)
        X_val = np.expand_dims(X_val, axis=4)

        train_data = X_train, y_train
        val_data = X_val, y_val

        return train_data, val_data


    # pad images with zeros so that we can 
    # cut same dimension cubes for edge cases

    def pad_zeros(self, X, margin=10):
        newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
        x_offset = margin
        y_offset = margin
        newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
        return newX
        

    # cut data cubes for labeled data in order to preserve 
    # both spatial and spectral information.
    # self.data_spatial_size is the spatial cube size

    def create_train_cube_data(self, X, y, remove_zeros=True):
        margin = int((self.data_spatial_size - 1) / 2)

        total_Xs = list()
        total_ys = list()

        for xi, yi in zip(X, y):

            xi_zero_padded = self.pad_zeros(xi, margin=margin)

            for c in range(margin, xi_zero_padded.shape[0] - margin):
                for r in range(margin, xi_zero_padded.shape[1] - margin):
                    
                    curr_label = yi[c-margin, r-margin]

                    if curr_label > 0:
                        
                        cube = xi_zero_padded[c - margin : c + margin + 1, r - margin : r + margin + 1]   
                        total_Xs.append(cube)
                        total_ys.append(curr_label - 1)


        total_Xs = np.array(total_Xs)
        total_ys = np.array(total_ys)

        return total_Xs, total_ys


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

        pca = PCA(n_components=self.pca_components, whiten=True)
        pca_X = pca.fit_transform(total_X)
        
        # reconstruct original images to
        # preserve spatial dependancies

        offset = 0
        final_X = list()
        for xi in X:
            flat_dim = xi.shape[0] * xi.shape[1]
            xi_split = pca_X[offset: offset + flat_dim]
            xi_reduced = np.reshape(xi_split, (xi.shape[0], xi.shape[1], self.pca_components))
            final_X.append(xi_reduced)
            offset += flat_dim

        return final_X


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

    # change labels to categorical and get next batch

    def next_batch(self, step = 'train', batch_size):

        if step == 'train':
            X_batch, y_batch = self.train_data[self.train_offset : self.train_offset + batch_size]
            batch_size = X_batch.shape[0]
            self.train_offset += batch_size
        else:
            X_batch, y_batch = self.val_data[self.val_offset : self.val_offset + batch_size]
            batch_size = X_batch.shape[0]
            self.val_offset += batch_size

        y_batch = np_utils.to_categorical(y_batch)
        return X_batch, y_batch

    # reset offset after train epoch
    def reset_train_offset(self):
        self.train_offset = 0
    
    # reset offset after val epoch
    def reset_val_offset(self):
        self.val_offset = 0

            

        
TrainDataLoader(pca_components=30)
