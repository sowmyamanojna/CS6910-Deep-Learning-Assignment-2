#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import wandb
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from keras_tqdm import TQDMCallback


# ## Preparing training and validation sets 

# In[2]:


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
              interpolation = 'bilinear')


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
              subset = 'validation',
              interpolation = 'bilinear')

# Rescaling the data
scaler = Rescaling(1.0/255)
train_iter = train_data.map(lambda x, y: (scaler(x), y))
val_iter = val_data.map(lambda x, y: (scaler(x), y))


# In[3]:


# Preparing training and validation sets with augmentation
data_augmenter = ImageDataGenerator(
    rescale = 1.0/255,
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    brightness_range = [0.2, 1.5],
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip=True,
    data_format = 'channels_last',
    validation_split = 0.2)

data_dir = 'nature_12K/inaturalist_12K/train'
train_aug_iter = data_augmenter.flow_from_directory(data_dir, shuffle = True,                                                           seed = 17, subset = 'training')
val_aug_iter = data_augmenter.flow_from_directory(data_dir, shuffle = True,                                                   seed = 17, subset = 'validation')


# ## Setting up wandb sweeps

# In[4]:


sweep_config = {'name': 'random-test-sweep', 'method': 'grid'}
sweep_config['metric'] = {'name': 'val_acc', 'goal': 'maximize'}
parameters_dict = {
                   'first_layer_filters': {'values': [32, 64]},
                   'filter_org': {'values': [0.5, 1, 2]}, # Halving, same, doubling in subsequent layers
                   'data_aug': {'values': [False, True]},
                   'batch_norm': {'values': [False, True]}, 
                   'dropout': {'values': [0.0, 0.2, 0.3]},
                   'kernel_size': {'values': [3, 5]},
                   'dense_size': {'values': [50, 100]},
                   'activation': {'values': ['relu']},
                   'num_epochs': {'values': [10]}, 
                   'optimizer': {'values': ['adam']}, 
                  }
sweep_config['parameters'] = parameters_dict
#print(sweep_config)


# In[5]:


def CNN_wandb_sweeps(config = sweep_config):
    with wandb.init(config = config):
        config = wandb.init().config
        wandb.run.name = 'firstLayerFilters_{}_filterOrg_{}_dataAug_{}_batchNorm_{}_dropout_{}_kerSize_{}_denseSize_{}'                         .format(config.first_layer_filters, config.filter_org, config.data_aug,                                  config.batch_norm, config.dropout, config.kernel_size, config.dense_size)
        
        '''Defining the architecture'''
        inputs = tf.keras.Input(shape=(256, 256, 3))
        x = Rescaling(scale=1.0)(inputs)
        filter_sizes = [int(config.first_layer_filters*(config.filter_org**layer_num))                         for layer_num in range(5)]
        ker_size = config.kernel_size

        # Apply some convolution and pooling layers
        for layer_num in range(5):
            x = layers.Conv2D(filters = filter_sizes[layer_num], kernel_size=(ker_size, ker_size),                               activation = config.activation)(x)
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
            if config.batch_norm:
                x = layers.BatchNormalization(axis = -1)(x)


        # Dense Layer
        x = layers.Flatten()(x)
        if config.dropout > 0:
            x = layers.Dropout(rate = config.dropout)(x)
        x = layers.Dense(config.dense_size, activation = config.activation)(x)
        if config.batch_norm:
            x = layers.BatchNormalization(axis = -1)(x)

        # Output Layer
        outputs = layers.Dense(10, activation = config.activation)(x)
        model = tf.keras.Model(inputs = inputs, outputs = outputs)
        
        '''Fitting the model'''      
        # Using original or augmented dataset as specified
        sweep_train_iter = train_aug_iter if config.data_aug else train_iter
        sweep_val_iter = val_aug_iter if config.data_aug else val_iter
        
        model.compile(optimizer = config.optimizer,
                      loss = tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")])
        
        model_hist = model.fit(sweep_train_iter, epochs = config.num_epochs)
        train_loss, train_acc = model_hist.history['loss'][-1], model_hist.history['acc'][-1]
        
        # Evaluating on validation set
        val_loss, val_acc = model.evaluate(sweep_val_iter)
        
        # Logging in wandb
        wandb.log({'train_loss': train_loss, 'train_acc': train_acc,
                   'val_loss': val_loss, 'val_acc': val_acc})


# In[ ]:


os.environ['WANDB_NOTEBOOK_NAME'] = 'Part-A-wandb.ipynb'
sweep_id = wandb.sweep(sweep_config, project = 'DL-Assignment2-PartA-10')
wandb.agent(sweep_id, function = CNN_wandb_sweeps)


# In[ ]:




