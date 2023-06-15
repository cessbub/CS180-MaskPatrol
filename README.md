# CS180-MaskPatrol 🚀

## Creators 👩🏻‍💻
Ventures, Calderon, Calma, Tan 

## Project Summary 📝

The Philippines' pandemic is not yet over as there are still sudden surges of COVID-19 cases. However, the UP community is transitioning to a face-to-face (F2F) setup. In times like these, safety protocols should be strictly enforced since the virus spreads rapidly between people within close proximity. Some of these protocols and guidelines are to regularly wear face masks and to practice social distancing. 

This project proposal aims to solve this relevant and timely problem, especially for the UP community. Machine learning and computer vision will be used to develop a model that will determine in real-time whether a person is wearing a face mask correctly or incorrectly, and if social distancing is observed. 

The project will use datasets of images of what it wants to detect in order to create a machine model that will detect the different situations. Furthermore, Convolutional Neural Network will be used to train the model so that it can label the situations more accurately to be used by the community. The project will then implement the model so that it can label real-time videos using computer vision that will be deployed into a WebApp. Additionally, it aims to include an alarm system if a violation of the safety protocols was observed.

## Table of Contents 📑

- [How to run this repo](#how-to-run-this-repo)
- [Directory structure](#directory-structure)
- [Where to get the datasets](#where-to-get-the-datasets)

## How to run this repo ▶️

1. Git clone the repository
2. Create a virtual environment 
   - Run in terminal `python3 -m venv cv`
   - Run in terminal `cd cv`
   - Run in terminal `source bin/activate` 
3. Run in terminal
   - `conda install -c conda-forge opencv`
   - `pip install scipy`
   - `pip install tensorflow`
   - `pip install flask`
   - `pip install opencv-python`
4. Run in terminal `python face_detection_app.py`

## Directory structure 📁

```
CS180-MASKPATROL    
|__.venv
|__datasets
|       |__FMD-SD
|       |     |__Face Mask Detection
|       |             |__ annotations
|       |             |__ images
|       |__MaskedFace-Net + RMFD + 12k masks + custom images
|             |__NM (added more images)
|             |__CMFD (added more images)
|             |__IMFD (added more images)
|__face_detection_app.py
|__mask-detection.ipnyb
|__mask_detection.h5
|__README.me
```

## Where to get the datasets? 📂

1. [FMD-SD](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
2. [CMFD and IMFD](https://github.com/cabani/MaskedFace-Net)
3. [NM](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset) (_Note: Merged the WithoutMask folders into one NM folder_)

