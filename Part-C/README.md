## Part-C YOLO-v3 for Forest Fire Detection

The trained weights are taken from [OlafenwaMoses/FireNET](https://github.com/OlafenwaMoses/FireNET/).

The files required for execution are as follows:

- detection_model-ex-33--loss-4.97.h5
- detection_config.json
- pretrianed-yolov3.h5

In order to detect fire from the video `video1.mp4`, run the following:
```python
python3 detect_from_video
```

The fire dataset videos are taken from [MIVIA – Laboratorio di Macchine Intelligenti per il Riconoscimento di Video, Immagini e Audio](https://mivia.unisa.it/datasets/video-analysis-datasets/fire-detection-dataset/) and [Bilkent University](http://signal.ee.bilkent.edu.tr/VisiFire/).  

The actual and processed video files are as follows:

| Original file | Processed file       |
|---------------|----------------------|
| yolov3-1.mp4  | yolov3-detected-1.avi| 
| yolov3-2.mp4  | yolov3-detected-2.avi| 