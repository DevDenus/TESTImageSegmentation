import os
import numpy as np
import cv2

from datasets import Dataset, Features, Image
from tqdm import tqdm


ss_urbansyn_to_cityscapes_mapping = {
    0 : 7, 1 : 8, 2 : 11, 3 : 12,
    4 : 13, 5 : 17, 6 : 19, 7 : 20,
    8 : 21, 9 : 22, 10 : 23, 11 : 24,
    12 : 26, 13 : 27, 14 : 28, 15 : 31,
    16 : 32, 17 : 33, 18 : 0, 19 : 4
}

def preprocess_urbansyn_ss(ss_path : str, cache_dir : str = "./data/urbansyn/ss_prep/") -> np.array:
    """
    Preprocessing semantic segmentation of urbansyn dataset to the cityscapes form
    ss_path : str - path to semantic segmentation image in form of 'ss_n.png'
    cache_dir : str - path to directory result would be saved at
    """
    ss = np.array(cv2.imread(ss_path))

    for i in range(ss.shape[0]):
        for j in range(ss.shape[1]):
            ss[i, j, 0] = ss[i, j, 1] = ss[i, j, 2] = ss_urbansyn_to_cityscapes_mapping[int(ss[i,j, 0])]

    cv2.imwrite(os.path.join(cache_dir, ss_path.split('/')[-1]), ss)
    return ss

def preprocess_image_folder(rgb_path : str, ss_path : str, cache_dir : str = "./data") -> Dataset:
    """
    Makes a huggingface dataset out of image folder
    rgb_path : str - path to images in form of 'rgb_n.png'
    ss_path : str - path to semantic segmentation masks in form of 'ss_n.png'
        corresponding to 'rgb_n.png'
    """
    rgb_files = sorted([f for f in os.listdir(rgb_path) if f.endswith('.png')])
    ss_files = sorted([f for f in os.listdir(ss_path) if f.endswith('.png')])
    assert len(rgb_files) == len(ss_files), \
        f"Amount of images and corresponding masks must be the same! {len(rgb_files)} != {len(ss_files)}"

    dataset_data = []

    for rgb_file, ss_file in tqdm(zip(rgb_files, ss_files)):
        rgb_id = rgb_file.split('_')[1].split('.')[0]
        ss_id = ss_file.split('_')[1].split('.')[0]

        assert rgb_id == ss_id, \
            f"Files {rgb_file} and {ss_file} does not correspond to each other!"

        preprocess_urbansyn_ss(os.path.join(ss_path, ss_file), ss_path)

        dataset_data.append({
            'image' : os.path.join(rgb_path, rgb_file),
            'mask' : os.path.join(ss_path, ss_file)
        })

    features = Features({
        'image' : Image(),
        'mask' : Image()
    })

    dataset = Dataset.from_list(dataset_data).cast(features)

    dataset.save_to_disk(cache_dir)
    return dataset



if __name__ == "__main__":
    rgb_path = 'data/urbansyn/rgb'
    ss_path = 'data/urbansyn/ss'
    preprocess_image_folder(rgb_path, ss_path)
