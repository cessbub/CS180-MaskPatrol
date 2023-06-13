# CS180-MaskPatrol

**How to run this repo:**

1. Git clone the repository
2. Create a virtual environment <br>
  a. Run in terminal ```python3 -m venv cv``` <br>
  b. Run in terminal ```cd cv``` <br>
  c. Run in terminal ```source bin/activate``` 
3. Run in terminal ```conda install -c conda-forge opencv```
4. Run in terminal ```python face_detection_app.py```

**Directory structure**
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

**Where to get the datasets?**
1. [FMD-SD](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
2. [CMFD and IMFD](https://github.com/cabani/MaskedFace-Net)
3. [NM](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset) (_Note: Merged the WithoutMask folders into one NM folder_)
