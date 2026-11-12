"""DICOM -> inference -> stored result + overlay images."""
import io, logging, time
import numpy as np
import pydicom
from PIL import Image
from . import config, db, orthanc
from .engine import engine, ModelNotFound
from .gradcam import blend_heatmap, blend_mask

log = logging.getLogger("pipeline")
MIN_SLICES = 20


def _to_u8(arr, invert=False):
    lo, hi = np.percentile(arr, (0.5, 99.5))
    out = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    return ((1 - out if invert else out) * 255).astype(np.uint8)


def _xray_u8(ds):
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3:  # RGB secondary capture
        arr = arr.mean(-1)
    arr = arr * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
    return _to_u8(arr, invert=getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1")


def _z(ds):
    ipp = getattr(ds, "ImagePositionPatient", None)
    return float(ipp[2]) if ipp is not None else float(getattr(ds, "InstanceNumber", 0))


def _classify(ds0):
    gray = _xray_u8(ds0)
    out = engine.classify(gray)
    h, w = gray.shape
    cam = np.asarray(Image.fromarray((out["cam"] * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0
    probs = dict(sorted(zip(out["labels"], out["probs"]), key=lambda kv: -kv[1]))
    top = out["labels"][out["top"]]
    rec = {"kind": "classification", "top_label": top, "confidence": float(probs[top]),
           "details": {"predictions": probs, "heatmap_for": top}, "model_version": out["version"]}
    return rec, gray, blend_heatmap(gray, cam)


def _segment(datasets):
    ds = sorted(datasets, key=_z)
    stack = np.stack([d.pixel_array.astype(np.float32) * float(getattr(d, "RescaleSlope", 1))
                      + float(getattr(d, "RescaleIntercept", 0)) for d in ds])  # (z,y,x) HU
    zs = [_z(d) for d in ds]
    dz = float(np.median(np.abs(np.diff(zs)))) if len(zs) > 1 and np.ptp(zs) > 0 else float(getattr(ds[0], "SliceThickness", 2.0))
    ps = getattr(ds[0], "PixelSpacing", [1.0, 1.0])
    spacing = (float(ps[1]), float(ps[0]), dz or 2.0)  # (x, y, z)
    out = engine.segment(stack.transpose(2, 1, 0), spacing)
    mask_zyx = out["mask"].transpose(2, 1, 0)
    areas = mask_zyx.reshape(len(ds), -1).sum(1)
    k = int(areas.argmax()) if areas.max() > 0 else len(ds) // 2
    disp = _to_u8(np.clip(stack[k], -160, 240))
    voxels = int(mask_zyx.sum())
    rec = {"kind": "segmentation", "top_label": "Spleen" if voxels else "No structure found",
           "confidence": out["conf"], "model_version": out["version"],
           "details": {"voxels": voxels, "volume_ml": round(voxels * spacing[0] * spacing[1] * spacing[2] / 1000, 1),
                       "slice_index": k, "n_slices": len(ds)}}
    return rec, disp, blend_mask(disp, mask_zyx[k])


def analyze(datasets):
    """Pure function (no Orthanc I/O): list of pydicom datasets -> (record, preview u8, overlay rgb)."""
    ds0 = datasets[0]
    modality = str(getattr(ds0, "Modality", "")).upper()
    if modality == "CT":
        if len(datasets) < MIN_SLICES:
            raise ValueError(f"CT series needs at least {MIN_SLICES} slices for segmentation (got {len(datasets)}).")
        rec, prev, ov = _segment(datasets)
    elif modality == "MR":
        raise ValueError("MR is not supported: the segmentation model was trained on CT only.")
    else:
        rec, prev, ov = _classify(ds0)
    rec.update(modality=modality, study_uid=str(getattr(ds0, "StudyInstanceUID", "")))
    return rec, prev, ov


def process_blobs(orthanc_id, blobs):
    datasets = [pydicom.dcmread(io.BytesIO(b)) for b in blobs]
    rec, prev, ov = analyze(datasets)
    img_dir = config.DATA_DIR / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(prev).save(img_dir / f"{orthanc_id}_preview.png")
    Image.fromarray(ov).save(img_dir / f"{orthanc_id}_overlay.png")
    rec.update(orthanc_id=orthanc_id, status="pending", created=time.time())
    db.save_study(rec)
    return rec


def process_study(orthanc_id):
    try:
        ids = orthanc.study_instance_ids(orthanc_id)
        return process_blobs(orthanc_id, [orthanc.instance_file(i) for i in ids])
    except (ModelNotFound, ValueError, Exception) as e:  # record the failure so the UI can show it
        log.exception("processing %s failed", orthanc_id)
        db.save_study({"orthanc_id": orthanc_id, "status": "error", "error": str(e), "created": time.time()})
