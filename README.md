# ChexNet Model Compression

This app is a pneumonia detection application/model trainer. Dataset used for training is the [ISIC Archive](https://www.isic-archive.com/#!/topWithHeader/onlyHeaderTop/gallery?filter=%5B%5D) provided by ISIC (International Skin Imaging Collaboration).

##  Model Performance

Model was trained using the ISIS Archive dataset but due to limitations from the fact that dataset contained mostly white people with the condition, extra data of a thousand black people with melanoma was sourced and added to the dataset for training to ensure model is more robust.


## Requirements

*   Python 3.5 or newer.
*   Other requirements can be installed using the requirements.txt.
  

## Setup

Clone the project:
```
git clone https://github.com/Pucadopr/CheXNet-Model-Compression.git
```
Enter project directory:
```
cd CheXNet-Model-Compression
```
Downloading the dataset:
```
tar -xvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
```
Install required modules:
```
pip3 install -r requirements.txt
```
Train model:
```
python3 main.py
```
