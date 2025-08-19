import streamlit as st
import torch
import nibabel as nib
import numpy as np
import tempfile
import matplotlib.pyplot as plt
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Compose, Orientationd,
    Spacingd, NormalizeIntensityd, ToTensord
)
from pathlib import Path
from src.models.neurosegnet import NeuroSegNet
from src.xai.gradcam import generate_3d_gradcam

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "experiments/neurosegnet_v1/best_model.pth"

# Define preprocessing for uploaded file 
@st.cache_data
def preprocess_nii(file_path):
    
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image"]),
    ])
    data = transforms({"image": file_path})
    return data["image"].unsqueeze(0)  # Add batch dim

# App UI 
st.title( "NeuroSegNet Tumor Segmentation System")

uploaded_file = st.file_uploader("Upload 4-channel NIfTI (.nii.gz) volume", type=["nii.gz"])
show_gradcam = st.checkbox("Show Grad-CAM", value=True)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    volume = preprocess_nii(tmp_path).to(DEVICE)  # [1, 4, D, H, W]
    st.write("Input shape:", volume.shape)

    # Load model
    model = NeuroSegNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        output = torch.sigmoid(model(volume))  # [1, 1, D, H, W]

    pred_np = output.cpu().numpy()[0, 0]
    mid_slice = pred_np.shape[2] // 2

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(volume[0, 0, :, :, mid_slice].cpu(), cmap="gray")
    ax[0].set_title("Input (T1 or FLAIR)")
    ax[1].imshow(pred_np[:, :, mid_slice], cmap="hot")
    ax[1].set_title("Prediction")
    st.pyplot(fig)

    # ==== Grad-CAM ====
    if show_gradcam:
        cam = generate_3d_gradcam(model, volume, target_layer="encoder.layers.3", device=DEVICE)
        cam_np = cam[0, 0].cpu().numpy()

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(volume[0, 0, :, :, mid_slice].cpu(), cmap="gray")
        ax[0].set_title("Original")
        ax[1].imshow(cam_np[:, :, mid_slice], cmap="jet", alpha=0.7)
        ax[1].set_title("Grad-CAM")
        st.pyplot(fig)