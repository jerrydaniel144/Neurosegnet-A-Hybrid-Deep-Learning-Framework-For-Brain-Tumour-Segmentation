from src.data.preprocessing import normalize_zscore, resample_image
import numpy as np
import nibabel as nib

def test_normalization_zscore():
    dummy = np.random.randn(128, 128, 128)
    normed = normalize_zscore(dummy)
    assert np.isclose(np.mean(normed), 0, atol=1e-1), "Z-score mean not near 0"
    assert np.isclose(np.std(normed), 1, atol=1e-1), "Z-score std not near 1"

def test_resample_image():
    # Create dummy NIfTI
    affine = np.eye(4)
    data = np.random.rand(64, 64, 64)
    img = nib.Nifti1Image(data, affine)
    resampled = resample_image(img, new_spacing=(1.0, 1.0, 1.0))

    assert isinstance(resampled, nib.Nifti1Image), "Resampled output is not a NIfTI image"