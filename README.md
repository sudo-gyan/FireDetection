# Smoother and faster reporting of forest fire

There are lots of way and algorithm in-order to detect fire. We have decided to use Deep Learning using YOLO models in order to get an accurate as well as an efficient detection of the forest fire.

This project is made in Python.

It consists of an API for detection and a CLI detector as well.

API detector can be helpful to connect several Cameras and do the processing at one place.
For testing and standalone applications CLI code can be used.

A predefined YOLO v3 model was retrained to detect fire using `imageai` module which uses `tensorflow` behind the scenes.

## Screenshots

![Test Image](https://github.com/sudo-gyan/FireDetection/raw/master/tests/image/test.jpg)
![Processed Image](https://github.com/sudo-gyan/FireDetection/raw/master/tests/image/test-processed.png)

## Videos

![Test Video](https://github.com/sudo-gyan/FireDetection/raw/master/tests/video/video.mp4)
![Processed Video](https://github.com/sudo-gyan/FireDetection/raw/master/tests/video/video-processed.avi)

## How To Run

Grab the [YOLO model](https://drive.google.com/file/d/1-ZroDaCrfJcf0OwBcppBKCEek4Eu3Mo3/view?usp=sharing) and put it in `models` directory.  

Install dependencies by:  

```bash
pip install -r requirements.txt
```

Execute the code by running the respective file in python.
