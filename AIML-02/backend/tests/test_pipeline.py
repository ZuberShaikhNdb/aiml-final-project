import io
import pydicom, pytest
from app import pipeline, sample


def _read(b):
    return pydicom.dcmread(io.BytesIO(b))


def test_xray_is_classified_with_heatmap():
    rec, prev, ov = pipeline.analyze([_read(sample.make_xray(1))])
    assert rec["kind"] == "classification" and rec["modality"] == "DX"
    assert 0 <= rec["confidence"] <= 1
    assert prev.shape == ov.shape[:2] and ov.shape[2] == 3
    assert list(rec["details"]["predictions"].values()) == sorted(rec["details"]["predictions"].values(), reverse=True)


def test_ct_volume_is_segmented():
    rec, prev, ov = pipeline.analyze([_read(b) for b in sample.make_ct(24)])
    assert rec["kind"] == "segmentation" and rec["details"]["n_slices"] == 24
    assert rec["details"]["volume_ml"] > 0


def test_short_ct_is_rejected():
    with pytest.raises(ValueError):
        pipeline.analyze([_read(b) for b in sample.make_ct(5)])
