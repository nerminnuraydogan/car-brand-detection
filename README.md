# Car Brand Detection using YOLOv4

![bentley](https://user-images.githubusercontent.com/46245117/144742161-31a5f101-9c1b-41ad-b625-d7ab8177fd55.gif)
![bmw2](https://user-images.githubusercontent.com/46245117/144742175-c0bf97d9-691c-4743-8817-a1ffb9b06672.gif)
![mercedes](https://user-images.githubusercontent.com/46245117/144742182-e4a523dd-e0ea-4109-be14-cb019d74a0c2.gif)

This project is developed to detect car brands in images and videos using [YOLOv4](https://github.com/AlexeyAB/darknet). The model is trained with 10 car brands' images using pretrained YOLOv4 model.

Change `path_name` variable in pyhton files to experiment with images and videos.

To run:
- `pip3 install -r requirements.txt`
- Download the [model weights](https://drive.google.com/file/d/1Nf4yVn1RzoCSev8CQeU27szYYk8KWFNE/view?usp=sharing) and put them in `weights` folder.
- To generate a car brand detection image on `data/chrysler.jpg`:
    ```
    python yolo_opencv.py
    ```
    A new image `chrysler_yolo4.jpg` will appear which has the bounding boxes of the cars in the image.

- To read from a video file and make predictions on `data/mercedes.mp4`:
    ```
    python vid_yolo_opencv.py
    ```
    This will start detecting car brands in that video, in the end, it'll save the resulting video to `output_video/mercedes.avi`

#### Class Names

Following car brands are used to detect in this project.

- Audi
- BMW
- Bentley
- Chrysler
- Ford
- Honda
- Hyundai
- Mercedes-Benz
- Nissan
- Toyota

#### Dataset

The dataset is reconstructed from the [Stanford AI Lab - Cars Dataset](https://www.kaggle.com/jessicali9530/stanford-cars-dataset) by preprocessing properly to make it convertible to YOLO format.

#### Demo

Watch the demo below.
[![Watch the demo](https://user-images.githubusercontent.com/46245117/144743208-77b24a43-0961-4c85-8c30-16d74c115874.PNG)](https://youtu.be/MkEgz57MWkw)



