import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

VAL_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])


class PatchDCT(nn.Module):
    def __init__(self, patch_size: int = 8, top_k: int = 32):
        super().__init__()
        self.p = patch_size
        self.k = top_k
        self._build_dct_basis(patch_size)

    def _build_dct_basis(self, N: int):
        basis = torch.zeros(N, N, N, N)
        for u in range(N):
            for v in range(N):
                cu = math.sqrt(1 / N) if u == 0 else math.sqrt(2 / N)
                cv = math.sqrt(1 / N) if v == 0 else math.sqrt(2 / N)
                for x in range(N):
                    for y in range(N):
                        basis[u, v, x, y] = (
                            cu * cv
                            * math.cos(math.pi * (2 * x + 1) * u / (2 * N))
                            * math.cos(math.pi * (2 * y + 1) * v / (2 * N))
                        )
        self.register_buffer("kernel", basis.view(N * N, 1, N, N))

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.p
        x_flat = x.view(B * C, 1, H, W)
        dct = F.conv2d(x_flat, self.kernel, stride=p)
        _, _, Hf, Wf = dct.shape
        dct = dct.view(B, C, p * p, Hf, Wf)
        dct = dct[:, :, :self.k, :, :]
        return dct.reshape(B, C * self.k, Hf, Wf)


class FreqStream(nn.Module):
    def __init__(self, in_channels: int = 96, out_dim: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1), nn.GELU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.GELU(), nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.GELU(), nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(512, out_dim)

    def forward(self, x):
        return self.proj(self.encoder(x).flatten(1))


class CrossModalAttention(nn.Module):
    def __init__(self, dim_s: int, dim_f: int, hidden: int = 512):
        super().__init__()
        self.q_s = nn.Linear(dim_s, hidden)
        self.k_f = nn.Linear(dim_f, hidden)
        self.v_f = nn.Linear(dim_f, hidden)
        self.q_f = nn.Linear(dim_f, hidden)
        self.k_s = nn.Linear(dim_s, hidden)
        self.v_s = nn.Linear(dim_s, hidden)
        self.scale  = hidden ** -0.5
        self.norm_s = nn.LayerNorm(hidden)
        self.norm_f = nn.LayerNorm(hidden)

    def forward(self, fs, ff):
        gate_sf = torch.sigmoid(
            (self.q_s(fs) * self.k_f(ff)).sum(dim=-1, keepdim=True) * self.scale
        )
        a_sf = gate_sf * self.v_f(ff)
        gate_fs = torch.sigmoid(
            (self.q_f(ff) * self.k_s(fs)).sum(dim=-1, keepdim=True) * self.scale
        )
        a_fs = gate_fs * self.v_s(fs)
        return self.norm_s(a_sf), self.norm_f(a_fs)


class DualStreamForensicNet(nn.Module):
    SPATIAL_DIM = 1792
    FREQ_DIM    = 512
    FUSION_DIM  = 512

    def __init__(self, num_classes: int = 1, patch_size: int = 8, top_k: int = 32):
        super().__init__()
        backbone = tvm.efficientnet_b4(weights=None)
        self.spatial_feat = nn.Sequential(*list(backbone.children())[:-2])
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.spatial_proj = nn.Linear(self.SPATIAL_DIM, self.FUSION_DIM)
        self.patch_dct    = PatchDCT(patch_size, top_k)
        self.freq_stream  = FreqStream(in_channels=3 * top_k, out_dim=self.FREQ_DIM)
        self.fusion       = CrossModalAttention(self.FUSION_DIM, self.FREQ_DIM, self.FUSION_DIM)
        self.head = nn.Sequential(
            nn.Linear(self.FUSION_DIM * 2, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        fs  = self.spatial_pool(self.spatial_feat(x)).flatten(1)
        fs  = self.spatial_proj(fs)
        dct = self.patch_dct(x)
        ff  = self.freq_stream(dct)
        fs_att, ff_att = self.fusion(fs, ff)
        return self.head(torch.cat([fs_att, ff_att], dim=1)).squeeze(1)


class InferenceEngine:
    def __init__(self, checkpoint_path: str):
        self.device = DEVICE
        self.model  = DualStreamForensicNet()
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        print(f"Model loaded from epoch {ckpt['epoch']}  (AUC {ckpt['best_auc']:.4f})")

    def preprocess(self, pil_image: Image.Image) -> torch.Tensor:
        return VAL_TRANSFORM(pil_image.convert("RGB")).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, tensor: torch.Tensor) -> float:
        logit = self.model(tensor)
        return torch.sigmoid(logit).item()
