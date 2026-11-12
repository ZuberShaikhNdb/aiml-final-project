"""Grad-CAM for torchvision DenseNet + overlay helpers (numpy only for the overlay part)."""
import numpy as np


def jet(x):
    x = np.clip(x, 0, 1)
    return np.stack([np.clip(1.5 - np.abs(4 * x - 3), 0, 1),
                     np.clip(1.5 - np.abs(4 * x - 2), 0, 1),
                     np.clip(1.5 - np.abs(4 * x - 1), 0, 1)], -1)


def blend_heatmap(gray_u8, cam, alpha=0.55):
    rgb = np.stack([gray_u8] * 3, -1).astype(np.float32) / 255
    w = (alpha * np.clip(cam * 1.5, 0, 1))[..., None]
    return (((1 - w) * rgb + w * jet(cam)) * 255).astype(np.uint8)


def blend_mask(gray_u8, mask, color=(255, 60, 60), alpha=0.45):
    rgb = np.stack([gray_u8] * 3, -1).astype(np.float32)
    m = mask.astype(bool)
    rgb[m] = (1 - alpha) * rgb[m] + alpha * np.array(color, dtype=np.float32)
    return rgb.astype(np.uint8)


def gradcam_densenet(model, x, class_idx=None):
    """Return (sigmoid probs, class index used, cam in [0,1] at input resolution)."""
    import torch
    import torch.nn.functional as F
    model.eval()
    with torch.enable_grad():
        acts = F.relu(model.features(x))
        acts.retain_grad()
        logits = model.classifier(F.adaptive_avg_pool2d(acts, 1).flatten(1))
        probs = torch.sigmoid(logits)[0]
        idx = int(probs.argmax()) if class_idx is None else class_idx
        model.zero_grad()
        logits[0, idx].backward()
    w = acts.grad.mean((2, 3), keepdim=True)
    cam = F.relu((w * acts).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return probs.detach().cpu().numpy(), idx, cam.detach().cpu().numpy()
