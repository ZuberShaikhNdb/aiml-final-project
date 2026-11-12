"""Fine-tune DenseNet121 (ImageNet transfer learning) for multi-label chest X-ray classification.
python ml/train_classifier.py --csv data/processed/nih_labels.csv --epochs 5"""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T

META = {"path", "split", "Patient ID"}
MEAN, STD, SIZE = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 224  # keep in sync with backend/app/config.py


class CXR(Dataset):
    def __init__(self, df, labels, tf):
        self.p, self.y, self.tf = df["path"].tolist(), df[labels].values.astype("float32"), tf

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        return self.tf(Image.open(self.p[i]).convert("RGB")), torch.from_numpy(self.y[i])


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    P, Y = [], []
    for x, y in loader:
        P.append(torch.sigmoid(model(x.to(dev))).cpu().numpy())
        Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)


def aucs(P, Y, labels):
    out = {}
    for i, l in enumerate(labels):
        if 0 < Y[:, i].sum() < len(Y):
            out[l] = float(roc_auc_score(Y[:, i], P[:, i]))
    return out, float(np.mean(list(out.values())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="weights")
    ap.add_argument("--reports-dir", default="reports")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(a.csv)
    labels = [c for c in df.columns if c not in META]
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    norm = T.Normalize(MEAN, STD)
    train_tf = T.Compose([T.Resize((SIZE, SIZE)), T.RandomAffine(7, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                          T.ColorJitter(0.2, 0.2), T.ToTensor(), norm])
    eval_tf = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor(), norm])
    mk = lambda d, tf, sh: DataLoader(CXR(d, labels, tf), a.batch_size, shuffle=sh, num_workers=a.workers, pin_memory=True)
    tl, vl, el = mk(tr, train_tf, True), mk(va, eval_tf, False), mk(te, eval_tf, False)

    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, len(labels))
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    amp = dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    out = Path(a.out_dir); out.mkdir(exist_ok=True, parents=True)
    rep = Path(a.reports_dir); rep.mkdir(exist_ok=True, parents=True)
    best, history = -1.0, []

    for ep in range(a.epochs):
        model.train(); t0, run = time.time(), 0.0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            with torch.autocast(dev, enabled=amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += loss.item() * len(x)
        sched.step()
        P, Y = predict(model, vl, dev)
        _, val_auc = aucs(P, Y, labels)
        history.append({"epoch": ep + 1, "train_loss": run / len(tr), "val_mean_auc": val_auc})
        print(f"epoch {ep+1}/{a.epochs} loss={run/len(tr):.4f} val_auc={val_auc:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if val_auc > best:
            best = val_auc
            torch.save(model.state_dict(), out / "classifier.pt")

    model.load_state_dict(torch.load(out / "classifier.pt", map_location=dev))
    P, Y = predict(model, el, dev)
    per_class, test_auc = aucs(P, Y, labels)
    (out / "labels.json").write_text(json.dumps(labels))
    metrics = {"dataset": "NIH ChestX-ray14 (subset)", "model": "DenseNet121 (ImageNet init)",
               "n_train": len(tr), "n_val": len(va), "n_test": len(te), "epochs": a.epochs, "seed": a.seed,
               "best_val_mean_auc": best, "test_mean_auc": test_auc, "test_auc_per_class": per_class,
               "history": history, "device": dev, "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (rep / "classifier_metrics.json").write_text(json.dumps(metrics, indent=2))
    md = ["| Finding | Test AUC |", "|---|---|"] + [f"| {k} | {v:.3f} |" for k, v in per_class.items()]
    md.append(f"| **Mean** | **{test_auc:.3f}** |")
    (rep / "classifier_metrics.md").write_text("\n".join(md))
    print("test mean AUC:", round(test_auc, 4))


if __name__ == "__main__":
    main()
