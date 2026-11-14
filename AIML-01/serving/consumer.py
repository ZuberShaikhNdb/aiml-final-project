"""Kafka consumer: reads transactions and calls the scoring API for each one."""
import argparse, json, logging, os, time
import requests

log = logging.getLogger("consumer")


def handle(raw: bytes, api: str, session=None):
    """Score one Kafka message. Returns the API response, or None if the message was bad."""
    s = session or requests
    try:
        msg = json.loads(raw)
    except ValueError:
        log.warning("dropping non-JSON message")
        return None
    r = s.post(api.rstrip("/") + "/score", json=msg, timeout=30)
    if r.status_code == 422:  # malformed transaction: skip it, don't block the stream
        log.warning("rejected by API: %s", r.text)
        return None
    r.raise_for_status()
    return r.json()


def main():
    from confluent_kafka import Consumer
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", default=os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"))
    ap.add_argument("--topic", default=os.getenv("KAFKA_TOPIC", "transactions"))
    ap.add_argument("--api", default=os.getenv("API_URL", "http://localhost:8000"))
    a = ap.parse_args()
    c = Consumer({"bootstrap.servers": a.bootstrap, "group.id": "fraud-scorer", "auto.offset.reset": "earliest",
                  "enable.auto.commit": False})
    c.subscribe([a.topic])
    session, n = requests.Session(), 0
    logging.basicConfig(level=logging.INFO)
    while True:
        m = c.poll(1.0)
        if m is None or m.error():
            continue
        for attempt in range(5):  # retry transient API failures, then move on
            try:
                handle(m.value(), a.api, session)
                break
            except requests.RequestException as e:
                log.warning("API error (%s), retry %d", e, attempt + 1)
                time.sleep(2 ** attempt * 0.2)
        c.commit(m)
        n += 1
        if n % 1000 == 0:
            log.info("scored %d", n)


if __name__ == "__main__":
    main()
