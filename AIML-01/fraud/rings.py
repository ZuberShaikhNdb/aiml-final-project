"""Link alerts that share an account into 'fraud rings' (connected components, union-find)."""


def find_rings(alerts):
    """alerts: iterable of dicts with id, nameOrig, nameDest. Returns list of {ring_id, alert_ids, accounts}, size>=2."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in alerts:
        ra, rb = find(("a", a["nameOrig"])), find(("a", a["nameDest"]))
        parent[ra] = rb
    groups = {}
    for a in alerts:
        groups.setdefault(find(("a", a["nameOrig"])), []).append(a)
    rings = []
    for members in groups.values():
        if len(members) >= 2:
            accts = sorted({m["nameOrig"] for m in members} | {m["nameDest"] for m in members})
            rings.append({"alert_ids": sorted(m["id"] for m in members), "accounts": accts, "size": len(members)})
    rings.sort(key=lambda r: -r["size"])
    for i, r in enumerate(rings, 1):
        r["ring_id"] = i
    return rings
