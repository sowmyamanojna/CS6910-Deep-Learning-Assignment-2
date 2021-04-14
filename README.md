# CS6910 Assignment 2
Assignment 2 submission for the course CS6910 Fundamentals of Deep Learning.  

Team members: N Sowmya Manojna (BE17B007), Shubham Kashyapi (MM16B027)

---
## Part-A Training a Smaller Network from Scratch
### 1. Loading data
################################################################
# Preparing training (without augmentation) and validation set 
################################################################
```python
data_dir = pathlib.Path('/content/drive/MyDrive/inaturalist_12K/train') # Set path to the right directory
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

len_train, len_val = len(train_data), len(val_data)
train_data = train_data.take(int(0.25*len_train))
val_data = val_data.take(int(0.25*len_val))
```

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
