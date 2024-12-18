from datasets import DatasetDict, Dataset, load_from_disk, load_dataset, concatenate_datasets

def datasetdict_to_dataset(ds : DatasetDict) -> Dataset:
    """
    Makes a Dataset instance out of DatasetDict instance.
    ds : DatasetDict - dataset dict to convert
    """
    ds_keys = ds.keys()
    result = concatenate_datasets([
        ds[key] for key in ds_keys
    ])
    return result

def merge_datasets(dataset_list : list[Dataset], train_size : float = 0.7, test_size : float = 0.2, cache_dir : str = "./data/urbansyncityscapes", random_seed : int = 42) -> DatasetDict:
    """
    Merges given datasets into one, shuffling their data and splitting it. Saves the resulting DatasetDict
    dataset_list : list[Dataset] - list of datasets in form of huggingface dataset
    train_size : float - specifies part of dataset to be used for training
    test_size : float - specifies part of dataset to be used for testing.
        If train_size + test_size < 1 then the remaining part will be splitted into validation
    cache_dir : str - path to directory where DatasetDict would be saved at
    random_seed : int - random_seed for dataset shuffling
    """
    assert min(train_size, test_size) >= 0.0 and train_size + test_size <= 1.0, \
        "Dataset cannot be split according to specified train, test and validation size"
    val_size = 1 - train_size - test_size

    for ds in dataset_list:
        if isinstance(ds, DatasetDict):
            ds = datasetdict_to_dataset(ds)

    combined_ds = concatenate_datasets(dataset_list)
    shuffled_ds = combined_ds.shuffle(seed=random_seed)

    total_size = len(shuffled_ds)
    train_split = int(total_size * train_size)
    val_split = int(total_size * val_size)

    train_ds = shuffled_ds.select(range(0, train_split))
    val_ds = shuffled_ds.select(range(train_split, train_split + val_split))
    test_ds = shuffled_ds.select(range(train_split + val_split, total_size))

    splitted_ds = DatasetDict({
        "train" : train_ds,
        "validation" : val_ds,
        "test" : test_ds
    })

    splitted_ds.save_to_disk(cache_dir)
    return splitted_ds

if __name__ == "__main__":
    cityscapes_ds_train = load_dataset("Chris1/cityscapes", cache_dir="data/cityscapes", split='train')
    cityscapes_ds_val = load_dataset("Chris1/cityscapes", cache_dir="data/cityscapes", split='validation')
    # cityscape's test split is unlabeled, so it wouldn't be used for training or evaluation
    urbansyn_ds = load_from_disk("./data/urbansyn_ds")
    merge_datasets([cityscapes_ds_train, cityscapes_ds_val, urbansyn_ds])
