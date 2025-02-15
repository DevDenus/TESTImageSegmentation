import os
import zipfile

from urllib.request import urlretrieve
from datasets import load_dataset, load_from_disk
from src.image_preprocessing import preprocess_image_folder
from src.merge_datasets import merge_datasets

def main():
    os.makedirs('./data', exist_ok=True)
    os.makedirs('./data/urbansyn', exist_ok=True)

    urlretrieve('https://datasets.cvc.uab.es/urbansyn/rgb.zip', './data/urbansyn/rgb.zip')
    with zipfile.ZipFile('./data/urbansyn/rgb.zip', 'r') as zip_rgb:
        os.makedirs('./data/urbansyn/rgb', exist_ok=True)
        zip_rgb.extractall('./data/urbansyn/rgb')

    urlretrieve('https://datasets.cvc.uab.es/urbansyn/ss.zip', './data/urbansyn/ss.zip')
    with zipfile.ZipFile('./data/urbansyn/ss.zip', 'r') as zip_ss:
        os.makedirs('./data/urbansyn/ss', exist_ok=True)
        zip_ss.extractall('./data/urbansyn/ss')

    rgb_path = './data/urbansyn/rgb'
    ss_path = './data/urbansyn/ss'
    preprocess_image_folder(rgb_path, ss_path)

    cityscapes_ds_train = load_dataset("Chris1/cityscapes", cache_dir="data/cityscapes", split='train')
    cityscapes_ds_val = load_dataset("Chris1/cityscapes", cache_dir="data/cityscapes", split='validation')
    # cityscape's test split is unlabeled, so it wouldn't be used for training or evaluation
    urbansyn_ds = load_from_disk("./data/urbansyn_ds")
    merge_datasets([cityscapes_ds_train, cityscapes_ds_val, urbansyn_ds])

if __name__ == "__main__":
    main()
