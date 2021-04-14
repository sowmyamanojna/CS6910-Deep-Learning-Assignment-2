## Part-A Training a Smaller Network from Scratch
### 1) Q1 to Q3 (Model architecture and wandb sweeps)
Part-A-Q1-to-Q3.ipynb loads the training and validation datasets. The different hyperparameter configurations for wandb are specified in the variable sweep_config. 
```python
sweep_config = {'name': 'random-test-sweep', 'method': 'random'}
sweep_config['metric'] = {'name': 'val_acc', 'goal': 'maximize'}
parameters_dict = {
                   'first_layer_filters': {'values': [32, 64]},
                   'filter_org': {'values': [0.5, 1, 2]}, # Halving, same, doubling in subsequent layers
                   'data_aug': {'values': [False, True]},
                   'batch_norm': {'values': [False, True]}, 
                   'dropout': {'values': [0.0, 0.2, 0.3]},
                   'kernel_size': {'values': [3,5,7]},
                   'dense_size': {'values': [32, 64, 128]},
                   'activation': {'values': ['relu']},
                   'num_epochs': {'values': [50]}, 
                   'optimizer': {'values': ['adam']},
                   'conv_layers': {'values': [5]}
                  }
sweep_config['parameters'] = parameters_dict
```
The function CNN_train defines the model architecture, trains the model and logs the metrics to wandb.

### 2) Q4 (Training without wandb, evaluating on test set, visualizing test images and filters)
PartA_Q4.ipynb can be used to train a model using the optimal hyperparameters obtained in Q2 and Q3. The code for loading the data, initializing the model architecture and training is similar to the previous notebook. So this notebook can be used to quickly test the code without setting up the wandb sweeps. The optimal hyperparameters are specified as follows. These can be modified for the purpose of testing the code.
```python
optimal = {
              'first_layer_filters': 64,
              'filter_org': 1, # Same number of filters in all convolution layers
              'data_aug': True,
              'batch_norm': True, 
              'dropout': 0.2,
              'kernel_size': 3,
              'dense_size': 128,
              'activation': 'relu',
              'num_epochs': 50, 
              'optimizer': 'adam',
              'conv_layers': 5
          }
```
The trained model can also be accessed directly at https://drive.google.com/drive/folders/1343Tk13X9iyIxdF4SityFtGOV9soLS5c?usp=sharing
```python
model = tf.keras.models.load_model(pathlib.Path('/content/drive/MyDrive/DL_Assignment2_PartA_Model'))
```
The code for evaluating this model on the test set (Q4 (a)), visualizing test images (Q4 (b)) and visualizing kernels (Q4 (c)) is also included in the same notebook.

### 2) Q5 (Guided backpropogation)
PartA_Q5.ipynb
The gradient for relu activation is redefined. The saved model is loaded and guided backpropogation is performed. The gradient images are visualized. 


