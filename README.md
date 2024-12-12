# TESTImageSegmentation
Implementation of image segmentation Swin Transformer inspired network learning algorithm using PyTorch

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
After run this script to combine both datasets into one, which will be used for training, testing and validating the model
```
python ./src/merge_datasets.py
```
You are ready to start the learning process!

### Datasets used for training
- [UrbanSyn/UrbanSyn dataset](https://huggingface.co/datasets/UrbanSyn/UrbanSyn)
- [Chris1/cityscapes dataset](https://huggingface.co/datasets/Chris1/cityscapes)

### Additional resources
- Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo - [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030)
