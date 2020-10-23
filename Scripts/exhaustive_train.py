from train import train_model
from utils import create_folder
import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True) 

if __name__ == '__main__':
    base_folder = 'checkpoints'
    create_folder(base_folder)

    spectral_dimensions = [20, 30, 40, 50, 60]
    spatial_dimensions = [11, 15, 19, 25, 29]

    for spec_dim in spectral_dimensions:
        for spatial_dim in spatial_dimensions:
            save_folder = '{}/model_size_{}_spec_{}'.format(
                base_folder, spatial_dim, spec_dim)
            train_model(spec_dim, spatial_dim, epochs = 50, save_folder=save_folder)
