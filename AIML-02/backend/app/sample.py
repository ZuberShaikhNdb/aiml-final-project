"""Synthetic DICOM generator for pipeline testing/demos. NOT medical data; never use for reported metrics.
   python -m app.sample --orthanc http://localhost:8042 --xray 2 --ct 1"""
import argparse, io
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

SC, CT_SOP = "1.2.840.10008.5.1.4.1.1.7", "1.2.840.10008.5.1.4.1.1.2"


def _ds(modality, sop_class, study, series, arr, rows_cols):
    meta = FileMetaDataset()
    sop = generate_uid()
    meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID = sop_class, sop
    meta.TransferSyntaxUID, meta.ImplementationClassUID = ExplicitVRLittleEndian, generate_uid()
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID, ds.StudyInstanceUID, ds.SeriesInstanceUID = sop_class, sop, study, series
    ds.Modality, ds.PatientName, ds.PatientID = modality, "SYNTHETIC^TEST", "SYN001"
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.Rows, ds.Columns = rows_cols
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit, ds.PixelRepresentation = 15, 0
    ds.PixelData = arr.astype(np.uint16).tobytes()
    return ds


def _bytes(ds):
    buf = io.BytesIO()
    pydicom.dcmwrite(buf, ds, enforce_file_format=True) if int(pydicom.__version__.split(".")[0]) >= 3 \
        else ds.save_as(buf, write_like_original=False)
    return buf.getvalue()


def make_xray(seed=0, size=256):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size] / size
    lungs = np.exp(-(((xx - .3) / .15) ** 2 + ((yy - .5) / .3) ** 2)) + np.exp(-(((xx - .7) / .15) ** 2 + ((yy - .5) / .3) ** 2))
    arr = 2500 - 1200 * lungs + rng.normal(0, 60, (size, size))
    ds = _ds("DX", SC, generate_uid(), generate_uid(), arr, (size, size))
    return _bytes(ds)


def make_ct(n_slices=24, size=96, seed=0):
    rng = np.random.default_rng(seed)
    study, series, out = generate_uid(), generate_uid(), []
    yy, xx = np.mgrid[:size, :size] / size
    for k in range(n_slices):
        body = (((xx - .5) / .42) ** 2 + ((yy - .5) / .32) ** 2) < 1
        organ = (((xx - .62) / .1) ** 2 + ((yy - .45) / .08) ** 2) < 1
        hu = np.where(body, 40, -1000) + np.where(organ, 60, 0) + rng.normal(0, 8, (size, size))
        ds = _ds("CT", CT_SOP, study, series, hu + 1024, (size, size))
        ds.RescaleSlope, ds.RescaleIntercept = 1, -1024
        ds.PixelSpacing, ds.SliceThickness = [1.5, 1.5], 2.0
        ds.ImagePositionPatient, ds.InstanceNumber = [0, 0, 2.0 * k], k + 1
        out.append(_bytes(ds))
    return out


if __name__ == "__main__":
    from . import orthanc, config
    p = argparse.ArgumentParser()
    p.add_argument("--orthanc", default=config.ORTHANC_URL)
    p.add_argument("--xray", type=int, default=1)
    p.add_argument("--ct", type=int, default=0)
    a = p.parse_args()
    config.ORTHANC_URL = a.orthanc.rstrip("/")
    for i in range(a.xray):
        orthanc.upload_instance(make_xray(i))
    for i in range(a.ct):
        for b in make_ct(seed=i):
            orthanc.upload_instance(b)
    print("Uploaded synthetic studies to", config.ORTHANC_URL)
