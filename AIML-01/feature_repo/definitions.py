from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Array, Float64, Int64

account = Entity(name="account", join_keys=["account_id"])
source = FileSource(path="data/account_state.parquet", timestamp_field="event_timestamp")

# Per-account running state, as origin (o_*) and as destination (d_*). Updated by the scoring service after every transaction.
account_stats = FeatureView(
    name="account_stats", entities=[account], ttl=timedelta(days=3650), source=source,
    schema=[Field(name=f"{r}_{k}", dtype=t) for r in ("o", "d")
            for k, t in (("n", Int64), ("sum", Float64), ("recent", Array(Int64)))],
)
