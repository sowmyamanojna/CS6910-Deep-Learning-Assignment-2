# CS6910 Assignment 2
Assignment 2 submission for the course CS6910 Fundamentals of Deep Learning.  

Team members: N Sowmya Manojna (BE17B007), Shubham Kashyapi (MM16B027)

---
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

### 2) Q4 (Evaluating on test set and visualizing filters)
PartA_Q4.ipynb can be used to train a model using the optimal hyperparameters obtained in Q2 and Q3. The code for loading the data, initializing the model architecture and training is similar to the previous notebook. But this notebook can be used to quickly test the code without setting up the wandb sweeps. The optimal hyperparameters are specified as follows. These can be modified for the purpose of testing the code.
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



## Part-B Using Pre-trained Models for Image Classification
### 1. Dataset
The dataset can be downloaded using the [`drive_dataset_check.ipynb`](https://github.com/sowmyamanojna/CS6910-Deep-Learning-Assignment-2/blob/main/drive_dataset_check.ipynb) code present in the main directory.

The code downloads the [iNaturalist dataset](https://github.com/sowmyamanojna/CS6910-Deep-Learning-Assignment-2/blob/main/drive_dataset_check.ipynb), unzips it and renames the images sequentially (for easy identification of missing images in case of network discrepancy).

The images are then loaded, split into training & validation sets. The image size is restricted to (256, 256) and all images that do not conform to the  specified size are automatically resized.

### 2. Models
The trained model is made modular by implementing a `CNN` class.  

Instances of the class can be created by specifying the base model, a flag determining whether the weights of the last few layers should be trained and an offset (i.e.) the number of layers from the end that have to be trained. The available options of the parameters are as follows:

- `base_model_name`: A string that specifies the base model that should be loaded
    + "InceptionV3"
    + "InceptionResNetV2"
    + "ResNet50"
    + "Xception"
- `tune`: A flag that determines whether the last few layers can be trained (default: False)
    + True
    + False
- `offset`: An positive integer that determines the number of layers from the end that should be trained. (Active only when the `tune` parameter is set to True; default: 20)

### 3. Train without wandb
```python
base_model_name = "InceptionV3"
tune = False
offset = 20

model = CNN(base_model_name, tune, offset)

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
```

### 4. Train with wandb
```python
sweep_config = {'name': 'random-test-sweep', 'method': 'grid'}
sweep_config['metric'] = {'name': 'val_acc', 'goal': 'maximize'}
parameters_dict = {
                   'base_model_name': {'values': ["InceptionV3", "InceptionResNetV2", "ResNet50", "Xception"]},
                   'tune': {'values': [False, True]},
                  }
sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project = 'DL-Assignment2-PartB-9April')
wandb.agent(sweep_id, function=pretrain_CNN_sweep)
```

### 5. Results
The results of the parameter sweep can be accessed here: [Wandb report Part-B](https://wandb.ai/cs6910-team/assignment-2/reports/CS6910-Assignment-2--Vmlldzo1NzQ1MTc#part-b-:-fine-tuning-a-pre-trained-model)

---
## Part-C YOLO-v3 for Forest Fire Detection
### 1. Model - Weights and `config.json`
The trained weights are taken from [OlafenwaMoses/FireNET](https://github.com/OlafenwaMoses/FireNET/).

The files required for execution are as follows:

- [detection_model-ex-33--loss-4.97.h5](https://github.com/OlafenwaMoses/FireNET/releases/download/v1.0/detection_model-ex-33--loss-4.97.h5)
- [detection_config.json](https://github.com/OlafenwaMoses/FireNET/releases/download/v1.0/detection_config.json)
- [pretrianed-yolov3.h5](https://github.com/OlafenwaMoses/ImageAI/releases/download/essential-v4/pretrained-yolov3.h5)

Note that the size of `pretrianed-yolov3.h5` and `detection_model-ex-33--loss-4.97.h5` is greater than 100 Mb. So, the files have to downloaded separately.

### 2. Process Video
In order to detect fire from the video `video1.mp4`, run the following:
```python
python3 detect_from_video
```


### 3. Dataset
The fire dataset videos are taken from [MIVIA – Laboratorio di Macchine Intelligenti per il Riconoscimento di Video, Immagini e Audio](https://mivia.unisa.it/datasets/video-analysis-datasets/fire-detection-dataset/) and [Bilkent University](http://signal.ee.bilkent.edu.tr/VisiFire/).  


### 4. Results
The final processed video can be accessed here: [final.mp4](https://github.com/sowmyamanojna/CS6910-Deep-Learning-Assignment-2/blob/main/Part-C/final.mp4)

The youtube link for the same: [Final Video](https://www.youtube.com/watch?v=W7NxfqgGzAE)

The actual and processed video files can be accessed here: [CS6910-YOLOv3](https://drive.google.com/drive/folders/1PP5FhP5_MpRucv7xq-RVAYLalWfyJNH1?usp=sharing)
