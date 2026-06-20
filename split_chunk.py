"""split_chunk.py <input.pkl> <out_prefix> — partition db_pair by hash(key)%2 into two files."""
import pickle
import sys
import time


def main():
    in_path, out_prefix = sys.argv[1], sys.argv[2]
    t0 = time.time()
    with open(in_path, "rb") as fh:
        d = pickle.load(fh)
    print(f"loaded {in_path} in {time.time()-t0:.1f}s, pair keys={len(d['db_pair']):,}")

    halves = [{}, {}]
    t0 = time.time()
    for h, lst in d["db_pair"].items():
        halves[hash(h) % 2][h] = lst
    print(f"split in {time.time()-t0:.1f}s -> {len(halves[0]):,} / {len(halves[1]):,}")

    for i in range(2):
        out = {"db_pair": halves[i],
               "db_single": d["db_single"] if i == 0 else {},
               "stats": d["stats"] if i == 0 else {}}
        path = f"{out_prefix}_{i}.pkl"
        t0 = time.time()
        with open(path, "wb") as fh:
            pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"dumped {path} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
