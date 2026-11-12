"""Model loading + inference. Torch/MONAI are imported lazily so DEMO_MODE and tests run without them."""
import hashlib, json
import numpy as np
from . import config
from .gradcam import gradcam_densenet


class ModelNotFound(RuntimeError):
    pass


def build_unet():
    from monai.networks.layers import Norm
    from monai.networks.nets import UNet
    return UNet(spatial_dims=3, in_channels=1, out_channels=2, channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2), num_res_units=2, norm=Norm.BATCH)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


class Engine:
    def __init__(self):
        self._cls = self._seg = None
        self.labels = config.NIH_LABELS
        self.cls_version = self.seg_version = None

    @staticmethod
    def device():
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- 2D chest X-ray classification ----------
    def _load_cls(self):
        if self._cls is not None:
            return
        import torch
        from torchvision import models
        w, lab = config.WEIGHTS_DIR / "classifier.pt", config.WEIGHTS_DIR / "labels.json"
        if not (w.exists() and lab.exists()):
            raise ModelNotFound(f"Missing {w} / {lab}. Train with ml/train_classifier.py first.")
        self.labels = json.loads(lab.read_text())
        m = models.densenet121(weights=None)
        m.classifier = torch.nn.Linear(1024, len(self.labels))
        m.load_state_dict(torch.load(w, map_location=self.device()))
        self._cls = m.to(self.device()).eval()
        self.cls_version = "densenet121-" + _sha(w)

    def classify(self, gray):
        """gray: HxW uint8. Returns dict(labels, probs, top, cam[224x224], version)."""
        if config.DEMO_MODE:
            return self._demo_cls(gray)
        import torch
        from PIL import Image
        self._load_cls()
        img = Image.fromarray(gray).convert("RGB").resize((config.IMG_SIZE,) * 2, Image.BILINEAR)
        a = (np.asarray(img, dtype=np.float32) / 255 - config.MEAN) / config.STD
        x = torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device())
        probs, idx, cam = gradcam_densenet(self._cls, x)
        return {"labels": self.labels, "probs": probs.tolist(), "top": idx, "cam": cam,
                "version": self.cls_version}

    def _demo_cls(self, gray):
        rng = np.random.default_rng(int(hashlib.md5(gray.tobytes()).hexdigest()[:8], 16))
        probs = rng.beta(1, 6, size=len(config.NIH_LABELS))
        n = config.IMG_SIZE
        cy, cx = rng.uniform(0.3, 0.7, 2) * n
        yy, xx = np.mgrid[:n, :n]
        cam = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (n * 0.15) ** 2)))
        return {"labels": config.NIH_LABELS, "probs": probs.tolist(), "top": int(probs.argmax()),
                "cam": cam, "version": "DEMO-UNTRAINED"}

    # ---------- 3D CT segmentation ----------
    def _load_seg(self):
        if self._seg is not None:
            return
        import torch
        w = config.WEIGHTS_DIR / "segmentation.pt"
        if not w.exists():
            raise ModelNotFound(f"Missing {w}. Train with ml/train_segmentation.py first.")
        m = build_unet()
        m.load_state_dict(torch.load(w, map_location=self.device()))
        self._seg = m.to(self.device()).eval()
        self.seg_version = "monai-unet3d-" + _sha(w)

    def segment(self, vol, spacing):
        """vol: (x,y,z) float HU in DICOM-natural order; spacing: (sx,sy,sz) mm.
        Returns dict(mask (x,y,z) bool, conf, version)."""
        if config.DEMO_MODE:
            return self._demo_seg(vol)
        import torch
        import torch.nn.functional as F
        from monai.inferers import sliding_window_inference
        self._load_seg()
        v = vol[::-1, ::-1, :]  # LPS-like -> RAS-like, as in MONAI training (Orientationd RAS)
        lo, hi = config.SEG_HU
        v = np.ascontiguousarray(np.clip((v - lo) / (hi - lo), 0, 1), dtype=np.float32)
        new = [max(1, int(round(s * sp / tg))) for s, sp, tg in zip(v.shape, spacing, config.SEG_SPACING)]
        t = F.interpolate(torch.from_numpy(v)[None, None], size=new, mode="trilinear",
                          align_corners=False).to(self.device())
        with torch.no_grad():
            logits = sliding_window_inference(t, config.SEG_ROI, 4, self._seg, overlap=0.5)
            prob = torch.softmax(logits, 1)[:, 1:2]
            prob = F.interpolate(prob, size=v.shape, mode="trilinear", align_corners=False)[0, 0]
        prob = prob.cpu().numpy()[::-1, ::-1, :]
        mask = prob > 0.5
        return {"mask": mask, "conf": float(prob[mask].mean()) if mask.any() else 0.0,
                "version": self.seg_version}

    def _demo_seg(self, vol):
        x, y, z = [np.arange(s) / s for s in vol.shape]
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        mask = ((X - .65) / .15) ** 2 + ((Y - .45) / .12) ** 2 + ((Z - .5) / .3) ** 2 < 1
        return {"mask": mask, "conf": 0.5, "version": "DEMO-UNTRAINED"}


engine = Engine()
