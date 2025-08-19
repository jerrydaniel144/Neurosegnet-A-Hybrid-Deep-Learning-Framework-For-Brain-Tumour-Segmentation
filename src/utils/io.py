import os
from glob import glob
from typing import List, Dict

def load_data_dicts(directory: str, ext: str = ".nii.gz") -> List[Dict]:
    """
    Loads 4D NIfTI volumes from the specified directory.
    Assumes filenames follow naming like '..._image.nii.gz' and '..._label.nii.gz'.
    """
    images = sorted(glob(os.path.join(directory, "*_image.nii.gz")))
    labels = sorted(glob(os.path.join(directory, "*_label.nii.gz")))

    if len(images) != len(labels):
        raise ValueError("Mismatch between number of images and labels")

    data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]
    return data_dicts