"""merge_two.py <pathA> <pathB> <outpath> — merge two {db_pair,db_single,stats} pickles."""
import pickle
import sys
import time
from collections import defaultdict


def main():
    a_path, b_path, out_path = sys.argv[1:4]
    t0 = time.time()
    with open(a_path, "rb") as fh:
        a = pickle.load(fh)
    with open(b_path, "rb") as fh:
        b = pickle.load(fh)
    print(f"loaded both in {time.time()-t0:.1f}s")

    t0 = time.time()
    db_p = defaultdict(list)
    for h, lst in a["db_pair"].items():
        db_p[h].extend(lst)
    for h, lst in b["db_pair"].items():
        db_p[h].extend(lst)

    db_s = defaultdict(list)
    for h, lst in a["db_single"].items():
        db_s[h].extend(lst)
    for h, lst in b["db_single"].items():
        db_s[h].extend(lst)

    stats = dict(a["stats"])
    stats.update(b["stats"])
    print(f"merged in {time.time()-t0:.1f}s -> pair keys {len(db_p):,}  single keys {len(db_s):,}")

    t0 = time.time()
    out = {"db_pair": dict(db_p), "db_single": dict(db_s), "stats": stats}
    with open(out_path, "wb") as fh:
        pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"dumped in {time.time()-t0:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
