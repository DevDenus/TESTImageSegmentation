# TESTImageSegmentation
Implementation of image segmentation network learning algorithm using PyTorch

### Getting started
First of all downloading UrbanSyn dataset and unzipping it
```
wget https://datasets.cvc.uab.es/urbansyn/rgb.zip -o ./data/urbansyn
```
```
wget https://datasets.cvc.uab.es/urbansyn/ss.zip -o ./data/urbansyn
```
```
unzip -d ./data/urbansyn ./data/urbansyn/rgb.zip
```
```
unzip -d ./data/urbansyn ./data/urbansyn/ss.zip
```
Installing all needed python libraries
```
pip install -r requirements.txt
```
Run this script in order to form a dataset, which is compatible with cityscapes dataset
```
python ./src/image_preprocessing.py
```
You are ready to start the learning process!

### Datasets used for training
- https://huggingface.co/datasets/UrbanSyn/UrbanSyn
- https://huggingface.co/datasets/Chris1/cityscapes
