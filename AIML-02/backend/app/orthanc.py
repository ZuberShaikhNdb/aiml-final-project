import requests
from . import config


def _get(path, **kw):
    r = requests.get(config.ORTHANC_URL + path, timeout=60, auth=_auth(), **kw)
    r.raise_for_status()
    return r


def _auth():
    return (config.ORTHANC_USER, config.ORTHANC_PASS) if config.ORTHANC_USER else None


def changes(since):
    return _get("/changes", params={"since": since, "limit": 100}).json()


def study_instance_ids(study_id):
    return [i["ID"] for i in _get(f"/studies/{study_id}/instances").json()]


def instance_file(instance_id):
    return _get(f"/instances/{instance_id}/file").content


def upload_instance(data: bytes):
    r = requests.post(config.ORTHANC_URL + "/instances", data=data, timeout=120, auth=_auth(),
                      headers={"Content-Type": "application/dicom"})
    r.raise_for_status()
    return r.json()
