#!/usr/bin/env python
# coding: utf-8
import os
import wandb
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from PIL import Image
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

####################################################################
# Preparing training set (without augmentation) and validation set
####################################################################

# Preparing training and validation sets without augmentation
# Loading data from directory
data_dir = 'nature_12K/inaturalist_12K/train'
train_data = tf.keras.preprocessing.image_dataset_from_directory(
                      directory = data_dir,
                      labels = 'inferred',  
                      label_mode = 'categorical',
                      color_mode = 'rgb',
                      batch_size = 32,
                      image_size = (256, 256),
                      shuffle = True,
                      seed = 17,
                      validation_split = 0.2,
                      subset = 'training')

val_data = tf.keras.preprocessing.image_dataset_from_directory(
                      directory = data_dir,
                      labels = 'inferred',  
                      label_mode = 'categorical',
                      color_mode = 'rgb',
                      batch_size = 32,
                      image_size = (256, 256),
                      shuffle = True,
                      seed = 17,
                      validation_split = 0.2,
                      subset = 'validation')

# Retaining 25 percent of train and validation data and discarding the rest
train_data = train_data.take(int(len(train_data)*0.25))
val_data = val_data.take(int(len(val_data)*0.25))

################################################################
# Preparing training set with augmentation 
################################################################
# Using half of currently available training data (1/8 th of original training data) for augmentation 
train_data_reduced = train_data.take(int(0.5*len(train_data)))

data_augment_gen = ImageDataGenerator(
                    rescale = 1.0,
                    rotation_range = 20,
                    width_shift_range = 0.2,
                    height_shift_range = 0.2,
                    brightness_range = [0.2, 1.5],
                    shear_range = 0.2,
                    zoom_range = 0.2,
                    horizontal_flip=True,
                    data_format = 'channels_last')

def data_augment(image, label):
    data_augment_gen.fit(image)
    aug_images = [image] 
    aug_labels = [label]
    
    num_augment = 1
    aug_images.extend([next(data_augment_gen.flow(image)) for _ in range(num_augment)])
    aug_labels.extend([label for _ in range(num_augment)])
    
    return aug_images, aug_labels

train_data_aug = train_data_reduced.map(lambda image, label: 
                     tf.numpy_function(data_augment, [image, label], [tf.float32, tf.float32]))

###############################################
# Listing the hyperparameters in wandb config 
###############################################
sweep_config = {'name': 'random-test-sweep', 'method': 'random'}
sweep_config['metric'] = {'name': 'val_acc', 'goal': 'maximize'}
parameters_dict = {
                   'first_layer_filters': {'values': [32, 64]},
                   'filter_org': {'values': [0.5, 1, 2]}, # Halving, same, doubling in subsequent layers
                   'data_aug': {'values': [False, True]},
                   'batch_norm': {'values': [False, True]}, 
                   'dropout': {'values': [0.0, 0.2, 0.3]},
                   'kernel_size': {'values': [3, 5, 7]},
                   'dense_size': {'values': [32, 64, 128]},
                   'activation': {'values': ['relu']},
                   'num_epochs': {'values': [50]}, 
                   'optimizer': {'values': ['adam']},
                   'conv_layers': {'values': [5]}
                  }
sweep_config['parameters'] = parameters_dict

#####################################
# Defining the train function
#####################################
def CNN_train(config=sweep_config):
    with wandb.init(config=config):
        config = wandb.init().config
        wandb.run.name = 'firstLayerFilters_{}_filterOrg_{}_dataAug_{}_batchNorm_{}_dropout_{}_kerSize_{}_denseSize_{}'                                               .format(config.first_layer_filters, config.filter_org, config.data_aug, config.batch_norm,
                           config.dropout, config.kernel_size, config.dense_size)
        
        ###########################################
        # Initializing the model architecture
        ###########################################
        inputs = tf.keras.Input(shape=(256, 256, 3))
        x = Rescaling(scale=1.0/255)(inputs)
        filter_sizes = [int(config.first_layer_filters*(config.filter_org**layer_num)) for layer_num in range(config.conv_layers)]
        ker_size = config.kernel_size

        # Apply some convolution and pooling layers
        for layer_num in range(config.conv_layers):
            x = layers.Conv2D(filters=filter_sizes[layer_num], kernel_size=(ker_size, ker_size))(x)
            if config.batch_norm:
                x = layers.BatchNormalization(axis=-1)(x)
            x = layers.Activation(config.activation)(x)
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
            
                
        # Dense Layer
        x = layers.Flatten()(x)
        if config.batch_norm:
            x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Activation(config.activation)(x)
        if config.dropout > 0:
            x = layers.Dropout(rate = config.dropout)(x)        

        # Output Layer
        outputs = layers.Dense(10, activation ='softmax')(x)
        model = tf.keras.Model(inputs = inputs, outputs = outputs)
        #print(model.summary())
        ####################################
        # Training and evaluating the model
        ####################################
        # Using training data with or without augmentation
        sweep_train_data = train_data_aug if config.data_aug else train_data
        
        model.compile(optimizer=config.optimizer,
                      loss = tf.keras.losses.CategoricalCrossentropy(name = 'loss'),
                      metrics = [tf.keras.metrics.CategoricalAccuracy(name = 'acc')])
        
        # Fitting the model and logging metrics (train_loss, train_acc, val_loss, val_acc) after every epoch
        model_hist = model.fit(sweep_train_data, epochs = config.num_epochs,
                               validation_data = val_data, 
                               callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'val_acc', patience = 5),
                                            wandb.keras.WandbCallback()])

#################################
# Setting up wandb sweeps
#################################
sweep_id = wandb.sweep(sweep_config, project = 'DL-Assignment2-PartA-8April')
wandb.agent(sweep_id, function = CNN_train)

