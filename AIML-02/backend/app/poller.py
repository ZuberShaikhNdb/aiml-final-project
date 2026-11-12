import logging, threading
from . import config, db, orthanc, pipeline

log = logging.getLogger("poller")


def run(stop: threading.Event):
    while not stop.is_set():
        try:
            since = int(db.kv_get("changes_seq", "0"))
            while True:
                ch = orthanc.changes(since)
                for c in ch["Changes"]:
                    if c["ChangeType"] == "StableStudy":
                        pipeline.process_study(c["ID"])
                since = ch["Last"]
                db.kv_set("changes_seq", str(since))
                if ch.get("Done", True):
                    break
        except Exception as e:
            log.warning("Orthanc poll failed: %s", e)
        stop.wait(config.POLL_SECONDS)
