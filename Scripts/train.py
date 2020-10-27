from dataloader_train import TrainDataLoader
import tensorflow as tf
import os
from models import *
import numpy as np
from utils import create_folder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
import argparse

# train_step as tf function to boost performnce
@tf.function
def train_step(batch_xs, batch_ys, model, cat_loss, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(batch_xs, training=True)
        loss = cat_loss(batch_ys, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, predictions

# val_step as tf function to boost performace

@tf.function
def val_step(batch_xs, batch_ys, model, cat_loss):
    predictions = model(batch_xs, training=False)
    loss = cat_loss(batch_ys, predictions)
    return loss, predictions


# calculate aa from confusion matrix
def calc_average_acc(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return average_acc

# train epoch

def train_epoch(dataloader, model, batch_size, cat_loss, optimizer, train_loss,
                train_accuracy):

    batch_counter = 0

    # train epoch
    for batch_Xt, batch_yt in dataloader.batchify_data(batch_size, 'train'):
        loss, preds = train_step(batch_Xt, batch_yt, model, cat_loss,
                                 optimizer)

        train_loss(loss)
        train_accuracy(batch_yt, preds)

        batch_counter += 1

    # print trained epoch stats

    print('trained: {} batches'.format(batch_counter))

    print('train loss: {:.3f}, train acc: {:.2f}%'.format(
        train_loss.result(),
        train_accuracy.result() * 100))

# validate epoch

def val_epoch(dataloader,
              model,
              batch_size,
              cat_loss,
              val_loss,
              simple_flag=False):

    total_val_labs = list()
    total_val_preds = list()

    batch_counter = 0

    # validate and process predictions.
    # make them ready for validation metrics
    for batch_Xv, batch_yv in dataloader.batchify_data(batch_size, 'val'):
        loss, preds = val_step(batch_Xv, batch_yv, model, cat_loss)

        val_loss(loss)
        
        preds = np.argmax(preds, axis=1)
        total_val_labs.append(batch_yv)
        total_val_preds.append(preds)

        batch_counter += 1
    
    total_val_labs = np.concatenate(total_val_labs, axis=0)
    total_val_preds = np.concatenate(total_val_preds, axis=0)

    # calculate validation metrics

    oa = accuracy_score(total_val_labs, total_val_preds)
    confusion = confusion_matrix(total_val_labs, total_val_preds)
    aa = calc_average_acc(confusion)
    kappa = cohen_kappa_score(total_val_labs, total_val_preds)

    # print val epoch stats

    print('trained: {} val batches'.format(batch_counter))

    if simple_flag:
        print(
            'labeled val loss: {:.3f}, labeled val OA: {:.2f}%, labeled val AA: {:.2f}%, labeled Kappa: {:.2f}%'
            .format(val_loss.result(), oa * 100, aa * 100, kappa * 100))
    else:
        print(
            'val loss: {:.3f}, val OA: {:.2f}%, val AA: {:.2f}%, Kappa: {:.2f}%'
            .format(val_loss.result(), oa * 100, aa * 100, kappa * 100))

    return oa, aa, kappa

# train model based on input parameters

def train_model(spectral_components=30,
                spatial_size=25,
                epochs=100,
                batch_size=256,
                dataset_folder='.',
                save_folder='.',
                model_type='simple',
                train_mode='simple',
                learning_rate=0.001):

    # create save folder
    create_folder(save_folder)
    
    # create dataloader for train dataset
    dataloader = TrainDataLoader(pca_components=spectral_components,
                                 data_spatial_size=spatial_size,
                                 dataset_folder=dataset_folder,
                                 pca_save_folder=save_folder,
                                 train_mode=train_mode)

    # if train mode == 'euclidean' create dataloader for original data
    # so that we can validate on original labels only.
    # pca_load = true means pca is not trained again
    if train_mode != 'simple':
        simple_dataloader = TrainDataLoader(pca_components=spectral_components,
                                            data_spatial_size=spatial_size,
                                            pca_load=True,
                                            pca_save_folder=save_folder,
                                            train_mode='simple')


    # load model type
    if model_type == 'simple':
        model = model_3d2d(spatial_size, spectral_components)
    else:
        model = model_3d2d_residual(spatial_size, spectral_components)

    #setup optimizer
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate, decay=1e-06)

    #setup classification loss
    cat_loss = tf.keras.losses.SparseCategoricalCrossentropy()

    # create metrics for the losses
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_loss_simple = tf.keras.metrics.Mean(name='val_loss_simple')

    # create metric for train accuracy
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')

    # best model accuracy buffer
    save_max_acc = 0

    for epoch in range(epochs):
        
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_loss_simple.reset_states()

        print('Epoch {}'.format(epoch + 1))

        # train epoch
        train_epoch(dataloader, model, batch_size, cat_loss, optimizer,
                    train_loss, train_accuracy)

        # validate epoch
        oa, aa, kappa = val_epoch(dataloader, model, batch_size, cat_loss,
                                  val_loss)
        
        # if train_mode == 'euclidean'
        # validate again on original data
        if train_mode != 'simple':
            oa, aa, kappa = val_epoch(simple_dataloader,
                                      model,
                                      batch_size,
                                      cat_loss,
                                      val_loss_simple,
                                      simple_flag=True)

        # create total accuracy metric
        acc_value = oa * aa * kappa

        # save best model and metrics
        if acc_value > save_max_acc:
            save_max_acc = acc_value
            print('model saved with acc value = {:.2f}%'.format(acc_value *
                                                                100))
            model.save('{}/best_model.h5'.format(save_folder))

            with open('{}/metrics.txt'.format(save_folder), 'w') as f:
                print('val loss: {:.4f}'.format(val_loss.result()), file=f)
                print('val OA: {:.2f}%'.format(oa * 100, file=f), file=f)
                print('val AA: {:.2f}%'.format(aa * 100, file=f), file=f)
                print('val Kappa: {:.2f}%'.format(kappa * 100), file=f)

        # shuffle data for next epoch
        dataloader.shuffle_data()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', help='specify dataset_dir')
    parser.add_argument('--spatial_size', type=int, default=13, help='specify spatial size of the training data')
    parser.add_argument('--spectral_comps', type=int, default=30, help='specify spectral size of the trainign data')
    parser.add_argument('--batch_size', type=int, default=256, help='specify batch size')
    parser.add_argument('--epochs', type=int, default=100, help='specify number of epochs')

    args = parser.parse_args()

    if args.dataset_dir is None:
        print('You need to specify dataset_dir')
        exit(1)

    dataset_folder = args.dataset_dir

    spatial_size = args.spatial_size
    spectral_components = args.spectral_comps
    epochs = args.epochs
    batch_size = args.batch_size

    save_folder = 'model_size_{}_spec_{}'.format(spatial_size,
                                                 spectral_components)

    train_model(spectral_components,
                spatial_size,
                epochs,
                batch_size,
                dataset_folder,
                save_folder,
                model_type='resigual',
                train_mode='euclidean')
