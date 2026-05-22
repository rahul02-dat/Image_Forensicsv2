import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import base64
import io


class GradCAM:
    """GradCAM targeting the last block of EfficientNet-B4 spatial stream."""

    def __init__(self, model, target_layer=None):
        self.model  = model
        self.device = next(model.parameters()).device
        self._acts  = None
        self._grads = None

        layer = target_layer or model.spatial_feat[-1]
        self._fwd_hook = layer.register_forward_hook(self._save_acts)
        self._bwd_hook = layer.register_full_backward_hook(self._save_grads)

    def _save_acts(self, _, __, output):
        self._acts = output.detach()

    def _save_grads(self, _, __, grad_output):
        self._grads = grad_output[0].detach()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def generate(self, tensor: torch.Tensor) -> np.ndarray:
        """Returns a float32 heatmap in [0,1] at the input spatial resolution."""
        self.model.eval()
        tensor = tensor.requires_grad_(True)

        logit = self.model(tensor)
        self.model.zero_grad()
        logit.backward()

        weights = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        H = W = tensor.shape[-1]
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
        return cam.squeeze().cpu().float().numpy()


def heatmap_overlay(pil_image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> str:
    """Blend GradCAM heatmap onto the original image; return base64 PNG string."""
    img = np.array(pil_image.convert("RGB").resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_INFERNO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = (alpha * heatmap + (1 - alpha) * img).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()