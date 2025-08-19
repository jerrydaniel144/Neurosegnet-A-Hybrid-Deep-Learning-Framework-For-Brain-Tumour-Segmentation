import torch
import torch.nn.functional as F
import numpy as np

from monai.networks.utils import one_hot
from monai.transforms import Resize

class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class=None):
        output = self.model(input_tensor)
        if target_class is None:
            target_class = (output > 0.5).float()

        self.model.zero_grad()
        output.backward(gradient=target_class, retain_graph=True)

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3, 4])
        activations = self.activations[0]

        for i in range(len(pooled_gradients)):
            activations[i, :, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0)
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)
        return heatmap.cpu().numpy()


def apply_gradcam(model, input_tensor, target_layer, resize_to=(128, 128, 128)):
    gradcam = GradCAM3D(model, target_layer)
    cam = gradcam.generate_cam(input_tensor)

    # Resize CAM to match input dimensions if needed
    resize = Resize(spatial_size=resize_to)
    cam_tensor = torch.tensor(cam[None, None], dtype=torch.float32)
    cam_resized = resize(cam_tensor)
    return cam_resized[0, 0].numpy()