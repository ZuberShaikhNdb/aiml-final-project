"""Train a 3D U-Net (MONAI) on Medical Segmentation Decathlon Task09_Spleen (CT).
python ml/train_segmentation.py --data-dir data/raw/msd --epochs 100"""
import argparse, json, time
from pathlib import Path
import torch
from monai.apps import DecathlonDataset
from monai.data import DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import (AsDiscrete, Compose, CropForegroundd, EnsureChannelFirstd, EnsureTyped, LoadImaged,
                              Orientationd, RandCropByPosNegLabeld, ScaleIntensityRanged, Spacingd)
from monai.utils import set_determinism

SPACING, HU, ROI = (1.5, 1.5, 2.0), (-57, 164), (96, 96, 96)  # keep in sync with backend/app/config.py


def base(keys=("image", "label")):
    return [LoadImaged(keys), EnsureChannelFirstd(keys), Orientationd(keys, axcodes="RAS"),
            Spacingd(keys, pixdim=SPACING, mode=("bilinear", "nearest")),
            ScaleIntensityRanged(keys[0], a_min=HU[0], a_max=HU[1], b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys, source_key=keys[0])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw/msd")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="weights")
    ap.add_argument("--reports-dir", default="reports")
    a = ap.parse_args()
    set_determinism(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf = Compose(base() + [RandCropByPosNegLabeld(("image", "label"), label_key="label", spatial_size=ROI,
                                 pos=1, neg=1, num_samples=4, image_key="image", image_threshold=0),
                                 EnsureTyped(("image", "label"))])
    val_tf = Compose(base() + [EnsureTyped(("image", "label"))])
    kw = dict(root_dir=a.data_dir, task="Task09_Spleen", download=True, seed=a.seed, val_frac=0.2, cache_rate=0.0, num_workers=2)
    tr = DecathlonDataset(section="training", transform=train_tf, **kw)
    va = DecathlonDataset(section="validation", transform=val_tf, **kw)
    tl = DataLoader(tr, batch_size=2, shuffle=True, num_workers=2)
    vl = DataLoader(va, batch_size=1, num_workers=2)

    model = UNet(spatial_dims=3, in_channels=1, out_channels=2, channels=(16, 32, 64, 128, 256),
                 strides=(2, 2, 2, 2), num_res_units=2, norm=Norm.BATCH).to(dev)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    dice = DiceMetric(include_background=False, reduction="mean")
    post_p, post_l = AsDiscrete(argmax=True, to_onehot=2), AsDiscrete(to_onehot=2)
    out, rep = Path(a.out_dir), Path(a.reports_dir)
    out.mkdir(exist_ok=True, parents=True); rep.mkdir(exist_ok=True, parents=True)
    best, best_ep, history = -1.0, 0, []

    for ep in range(1, a.epochs + 1):
        model.train(); tot = 0.0
        for b in tl:
            x, y = b["image"].to(dev), b["label"].to(dev)
            opt.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); opt.step()
            tot += loss.item()
        row = {"epoch": ep, "train_loss": tot / len(tl)}
        if ep % a.val_every == 0 or ep == a.epochs:
            model.eval()
            with torch.no_grad():
                for b in vl:
                    x, y = b["image"].to(dev), b["label"].to(dev)
                    p = sliding_window_inference(x, ROI, 4, model)
                    dice(y_pred=[post_p(i) for i in decollate_batch(p)], y=[post_l(i) for i in decollate_batch(y)])
            d = dice.aggregate().item(); dice.reset()
            row["val_dice"] = d
            if d > best:
                best, best_ep = d, ep
                torch.save(model.state_dict(), out / "segmentation.pt")
        history.append(row); print(row, flush=True)

    metrics = {"dataset": "MSD Task09_Spleen (CT)", "model": "MONAI 3D UNet", "n_train": len(tr), "n_val": len(va),
               "epochs": a.epochs, "best_val_dice": best, "best_epoch": best_ep, "seed": a.seed,
               "note": "Only 41 labelled volumes exist, so validation Dice is used for both model selection and "
                       "reporting (no separate test set). Treat it as an optimistic estimate.",
               "history": history, "device": str(dev), "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (rep / "segmentation_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("best val Dice:", round(best, 4), "at epoch", best_ep)


if __name__ == "__main__":
    main()
