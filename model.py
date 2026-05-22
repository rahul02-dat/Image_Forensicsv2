import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

VAL_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])


# ── TTA transform variants used at inference ──────────────────────────────────
TTA_TRANSFORMS = [
    T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ]),
    T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ]),
    T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(p=1.0),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ]),
]


# ── Social-media compression helpers ─────────────────────────────────────────

def estimate_jpeg_quality(pil_img: Image.Image) -> int:
    """
    Heuristic JPEG quality estimate via quantization table if available.
    Falls back to 75 (mid-quality) when tables are absent (e.g. PNG input).
    """
    try:
        qtables = getattr(pil_img, "quantization", None)
        if qtables:
            avg_q = int(np.mean(list(qtables[0].values())))
            if avg_q <= 8:  return 95
            if avg_q <= 16: return 80
            if avg_q <= 24: return 70
            if avg_q <= 48: return 55
            return 40
    except Exception:
        pass
    return 75  # assume mid-quality if unknown


def social_media_preprocess(pil_img: Image.Image) -> Image.Image:
    """
    Detect if the image looks like it came through social-media compression
    (WhatsApp / Instagram / Telegram) and apply mild unsharp masking to
    partially recover high-frequency detail lost during recompression.

    This runs BEFORE the standard normalization transform so the model
    sees partially restored textures rather than blurred DCT artefacts.
    """
    quality = estimate_jpeg_quality(pil_img)
    img_np = np.array(pil_img.convert("RGB"), dtype=np.uint8)

    if quality < 80:
        # Unsharp mask: recovers edge / frequency detail destroyed by JPEG
        # sigma tuned so it doesn't introduce ringing on very low quality images
        sigma = max(0.8, 2.0 - quality / 80.0)   # 0.8–1.2 for q 55–79
        blur  = cv2.GaussianBlur(img_np, (0, 0), sigmaX=sigma)
        strength = min(0.5, (80 - quality) / 60.0)  # 0.0–0.5 proportional boost
        img_np = cv2.addWeighted(img_np, 1.0 + strength, blur, -strength, 0)
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)

    return Image.fromarray(img_np)


# ── Model components ──────────────────────────────────────────────────────────

class PatchDCT(nn.Module):
    """
    Splits image into non-overlapping patches, applies 2-D DCT per patch,
    then stacks the coefficient planes as a multi-channel feature map.

    Change vs original: we now skip the DC + 4 lowest-frequency coefficients
    (indices 0-3) and use the mid-frequency band (4 : 4+top_k).  Mid-frequency
    coefficients survive JPEG compression far better than the very lowest ones,
    yet still carry the GAN / diffusion fingerprints that the original top-k
    approach targeted.
    """

    DCT_SKIP = 4   # skip DC + 3 lowest AC coefficients

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.p
        x_flat = x.view(B * C, 1, H, W)
        dct = F.conv2d(x_flat, self.kernel, stride=p)
        _, _, Hf, Wf = dct.shape
        dct = dct.view(B, C, p * p, Hf, Wf)
        # Mid-frequency band: skip DC + lowest-freq, more robust to JPEG
        s = self.DCT_SKIP
        dct = dct[:, :, s:s + self.k, :, :]
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, fs: torch.Tensor, ff: torch.Tensor):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fs  = self.spatial_pool(self.spatial_feat(x)).flatten(1)
        fs  = self.spatial_proj(fs)
        dct = self.patch_dct(x)
        ff  = self.freq_stream(dct)
        fs_att, ff_att = self.fusion(fs, ff)
        return self.head(torch.cat([fs_att, ff_att], dim=1)).squeeze(1)


# ── Inference engine ──────────────────────────────────────────────────────────

class InferenceEngine:
    def __init__(self, checkpoint_path: str):
        self.device = DEVICE
        self.model  = DualStreamForensicNet()
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        print(f"Model loaded from epoch {ckpt['epoch']}  (AUC {ckpt['best_auc']:.4f})")

    def preprocess(self, pil_image: Image.Image) -> torch.Tensor:
        """Standard single-view preprocessing (used for GradCAM)."""
        img = social_media_preprocess(pil_image)
        return VAL_TRANSFORM(img.convert("RGB")).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, tensor: torch.Tensor) -> float:
        """Single-view prediction. Prefer predict_tta for production use."""
        logit = self.model(tensor)
        return torch.sigmoid(logit).item()

    @torch.no_grad()
    def predict_tta(self, pil_image: Image.Image, n_views: int = 3) -> float:
        """
        Test-Time Augmentation prediction.
        Averages sigmoid probabilities over `n_views` augmented crops/flips of
        the preprocessed image, reducing sensitivity to WhatsApp resampling.

        Args:
            pil_image: Raw PIL image (before any transform).
            n_views:   How many TTA views to average (default 3 — matches
                       TTA_TRANSFORMS length; set lower to trade accuracy for speed).
        Returns:
            P(AI-generated) in [0, 1].
        """
        img = social_media_preprocess(pil_image).convert("RGB")
        probs = []
        for tfm in TTA_TRANSFORMS[:n_views]:
            tensor = tfm(img).unsqueeze(0).to(self.device)
            logit  = self.model(tensor)
            probs.append(torch.sigmoid(logit).item())
        return float(sum(probs) / len(probs))