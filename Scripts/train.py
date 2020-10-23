from dataloader_train import TrainDataLoader
import tensorflow as tf
import os
from models import *
import numpy as np
from utils import create_folder

@tf.function
def train_step(batch_xs, batch_ys, model, cat_loss, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(batch_xs, training=True)
        loss = cat_loss(batch_ys, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, predictions


@tf.function
def val_step(batch_xs, batch_ys, model, cat_loss):
    predictions = model(batch_xs, training=False)
    loss = cat_loss(batch_ys, predictions)
    return loss, predictions


def train_model(spectral_components=30,
                spatial_size=25,
                epochs=100,
                batch_size=256,
                save_folder='.',
                learning_rate=0.001):

    create_folder(save_folder)

    dataloader = TrainDataLoader(pca_components=spectral_components,
                                 data_spatial_size=spatial_size,
                                 pca_save_folder=save_folder)

    model = model_3d2d(spatial_size, spectral_components)

    optimizer = tf.keras.optimizers.Adam(lr=learning_rate, decay=1e-06)

    cat_loss = tf.keras.losses.CategoricalCrossentropy()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(
        name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='test_loss')
    val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')

    save_max_acc = 0

    for epoch in range(epochs):

        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()

        print('Epoch {}'.format(epoch + 1))

        for batch_Xt, batch_yt in dataloader.batchify_data(
                batch_size, 'train'):
            loss, preds = train_step(batch_Xt, batch_yt, model, cat_loss,
                                     optimizer)

            train_loss(loss)
            train_accuracy(batch_yt, preds)

        print('train loss: {:.3f}, train acc: {:.2f}%'.format(
            train_loss.result(),
            train_accuracy.result() * 100))

        for batch_Xv, batch_yv in dataloader.batchify_data(batch_size, 'val'):
            loss, preds = val_step(batch_Xv, batch_yv, model, cat_loss)

            val_loss(loss)
            val_accuracy(batch_yv, preds)

        print('val loss: {:.3}, val acc: {:.2f}%'.format(
            val_loss.result(),
            val_accuracy.result() * 100))

        if val_accuracy.result() > save_max_acc:
            save_max_acc = val_accuracy.result()
            print('model saved with acc = {:.2f}%'.format(save_max_acc * 100))
            model.save('{}/best_model.h5'.format(save_folder))
            with open('{}/metrics.txt'.format(save_folder), 'w') as f:
                print('val acc: {}%'.format(save_max_acc * 100), file=f)

        dataloader.shuffle_data()


if __name__ == '__main__':
    spatial_size = 9
    spectral_components = 14
    epochs = 100
    batch_size = 512
    save_folder = 'model_size_{}_spec_{}'.format(spatial_size,
                                                 spectral_components)

    train_model(spectral_components, spatial_size, epochs, batch_size,
                save_folder)
