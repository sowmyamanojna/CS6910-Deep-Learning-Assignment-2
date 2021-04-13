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
