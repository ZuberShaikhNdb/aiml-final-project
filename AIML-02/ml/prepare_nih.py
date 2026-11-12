"""Build a labels CSV with a PATIENT-LEVEL train/val/test split from NIH ChestX-ray14.
python ml/prepare_nih.py --data-entry data/raw/nih/Data_Entry_2017.csv --img-root data/raw/nih \
       --out data/processed/nih_labels.csv --max-images 20000"""
import argparse, os
import pandas as pd

LABELS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
          "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-entry", required=True)
    p.add_argument("--img-root", required=True)
    p.add_argument("--out", default="data/processed/nih_labels.csv")
    p.add_argument("--max-images", type=int, default=20000, help="subsample for limited compute")
    a = p.parse_args()

    paths = {f: os.path.join(r, f) for r, _, fs in os.walk(a.img_root) for f in fs if f.endswith(".png")}
    df = pd.read_csv(a.data_entry)
    df = df[df["Image Index"].isin(paths)].copy()
    df["path"] = df["Image Index"].map(paths)
    for l in LABELS:
        df[l] = df["Finding Labels"].apply(lambda s, l=l: int(l in s.split("|")))
    # Split by patient so the same patient never appears in two splits (avoids leakage).
    bucket = df["Patient ID"].apply(lambda x: (int(x) * 2654435761) % 100)
    df["split"] = bucket.apply(lambda b: "train" if b < 70 else "val" if b < 80 else "test")
    if len(df) > a.max_images:
        df = df.sample(a.max_images, random_state=42)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df[["path", "split", "Patient ID"] + LABELS].to_csv(a.out, index=False)
    print(df["split"].value_counts().to_dict(), "->", a.out)


if __name__ == "__main__":
    main()
