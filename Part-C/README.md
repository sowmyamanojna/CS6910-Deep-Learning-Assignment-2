## Part-C YOLO-v3 for Forest Fire Detection

### Model - Weights and `config.json`
The trained weights are taken from [OlafenwaMoses/FireNET](https://github.com/OlafenwaMoses/FireNET/).

The files required for execution are as follows:

- [detection_model-ex-33--loss-4.97.h5](https://github.com/OlafenwaMoses/FireNET/releases/download/v1.0/detection_model-ex-33--loss-4.97.h5)
- [detection_config.json](https://github.com/OlafenwaMoses/FireNET/releases/download/v1.0/detection_config.json)
- [pretrianed-yolov3.h5](https://github.com/OlafenwaMoses/ImageAI/releases/download/essential-v4/pretrained-yolov3.h5)

Note that the size of `pretrianed-yolov3.h5` and `detection_model-ex-33--loss-4.97.h5` is greater than 100 Mb. So, the files have to downloaded separately.

### Process Video
In order to detect fire from the video `video1.mp4`, run the following:
```python
python3 detect_from_video
```


### Dataset
The fire dataset videos are taken from [MIVIA – Laboratorio di Macchine Intelligenti per il Riconoscimento di Video, Immagini e Audio](https://mivia.unisa.it/datasets/video-analysis-datasets/fire-detection-dataset/) and [Bilkent University](http://signal.ee.bilkent.edu.tr/VisiFire/).  


### Results
The final processed video can be accessed here: [final.mp4](https://github.com/sowmyamanojna/CS6910-Deep-Learning-Assignment-2/blob/main/Part-C/final.mp4)

The youtube link for the same: [Final Video](https://www.youtube.com/watch?v=W7NxfqgGzAE)

The actual and processed video files can be accessed here: [CS6910-YOLOv3](https://drive.google.com/drive/folders/1PP5FhP5_MpRucv7xq-RVAYLalWfyJNH1?usp=sharing)