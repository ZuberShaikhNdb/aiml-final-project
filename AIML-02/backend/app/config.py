import os
from pathlib import Path

ORTHANC_URL = os.getenv("ORTHANC_URL", "http://localhost:8042").rstrip("/")
ORTHANC_USER = os.getenv("ORTHANC_USER") or None
ORTHANC_PASS = os.getenv("ORTHANC_PASS") or None
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
WEIGHTS_DIR = Path(os.getenv("WEIGHTS_DIR", "../weights"))
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
DISABLE_POLLER = os.getenv("DISABLE_POLLER", "false").lower() == "true"
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "5"))

# Must match ml/train_classifier.py and ml/train_segmentation.py
IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SEG_SPACING = (1.5, 1.5, 2.0)
SEG_HU = (-57.0, 164.0)
SEG_ROI = (96, 96, 96)
NIH_LABELS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
              "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
              "Fibrosis", "Pleural_Thickening", "Hernia"]
DISCLAIMER = ("For education only. Not for clinical use. Outputs are not a diagnosis and "
              "must not be used to make medical decisions.")
